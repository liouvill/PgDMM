import warnings
import weakref

import torch
import pyro.ops.jit
from pyro.distributions.util import scale_and_mask
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.util import (
    check_fully_reparametrized,
    is_validation_enabled,
    torch_item,
)
from pyro.util import warn_if_nan
from torch.distributions.utils import _sum_rightmost
from pyro.distributions.util import scale_and_mask
import math
from numbers import Real


def _check_mean_field_requirement(model_trace, guide_trace):
    """
    Checks that the guide and model sample sites are ordered identically.
    This is sufficient but not necessary for correctness.
    """
    model_sites = [
        name
        for name, site in model_trace.nodes.items()
        if site["type"] == "sample" and name in guide_trace.nodes
    ]
    guide_sites = [
        name
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample" and name in model_trace.nodes
    ]
    assert set(model_sites) == set(guide_sites)
    if model_sites != guide_sites:
        warnings.warn(
            "Failed to verify mean field restriction on the guide. "
            "To eliminate this warning, ensure model and guide sites "
            "occur in the same order.\n"
            + "Model sites:\n  "
            + "\n  ".join(model_sites)
            + "Guide sites:\n  "
            + "\n  ".join(guide_sites)
        )


# Analytic ELBO for Gaussian distributions
class Analytic_ELBO(Trace_ELBO):
    
    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)
        if is_validation_enabled():
            _check_mean_field_requirement(model_trace, guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, _ = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            loss = loss + loss_particle / self.num_particles

        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0

        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    #elbo_particle = elbo_particle + model_site["log_prob_sum"]
                    elbo_particle = elbo_particle + log_prob(model_site["fn"], model_site["value"], model_site["scale"], model_site["mask"])
                else:
                    guide_site = guide_trace.nodes[name]
                    if is_validation_enabled():
                        check_fully_reparametrized(guide_site)

                    # use kl divergence if available, else fall back on sampling
                    try:
                        kl_qp = kl_analytic(guide_site["fn"], model_site["fn"])
                        kl_qp = scale_and_mask(
                            kl_qp, scale=guide_site["scale"], mask=guide_site["mask"]
                        )
                        if torch.is_tensor(kl_qp):
                            assert kl_qp.shape == guide_site["fn"].batch_shape
                            kl_qp_sum = kl_qp.sum()
                        else:
                            kl_qp_sum = (
                                kl_qp * torch.Size(guide_site["fn"].batch_shape).numel()
                            )
                        elbo_particle = elbo_particle - kl_qp_sum
                    except NotImplementedError:
                        entropy_term = guide_site["score_parts"].entropy_term
                        elbo_particle = (
                            elbo_particle
                            + model_site["log_prob_sum"]
                            - entropy_term.sum()
                        )

        # handle auxiliary sites in the guide
        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample" and name not in model_trace.nodes:
                assert guide_site["infer"].get("is_auxiliary")
                if is_validation_enabled():
                    check_fully_reparametrized(guide_site)
                entropy_term = guide_site["score_parts"].entropy_term
                elbo_particle = elbo_particle - entropy_term.sum()

        loss = -(
            elbo_particle.detach()
            if torch._C._get_tracing_state()
            else torch_item(elbo_particle)
        )
        surrogate_loss = -elbo_particle
        return loss, surrogate_loss


# Analytic KL-divergence for Gaussian distributions
def kl_gaussian(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def kl_analytic(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_gaussian(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


# Analytic log probability for Gaussian distributions
def log_prob_gaussian(p, value):
    # compute the variance
    var = (p.scale ** 2)
    log_scale = math.log(p.scale) if isinstance(p.scale, Real) else p.scale.log()
    return -((value - p.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


def log_prob(p, value, scale, mask):
    log_p = log_prob_gaussian(p.base_dist, value)
    log_p = _sum_rightmost(log_p, p.reinterpreted_batch_ndims)
    log_p = scale_and_mask(log_p, scale, mask).sum()
    return log_p
