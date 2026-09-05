from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class SAE:
    W_enc: object
    b_enc: object
    W_dec: object
    b_dec: object | None
    activation: str
    k: int | None
    d_model: int
    d_sae: int
    hook_layer: int
    hook_component: str
    repo_id: str = ""

    def encode(self, x):
        import torch
        z = x @ self.W_enc + self.b_enc
        if self.activation == "topk":
            assert self.k is not None
            vals, idx = torch.topk(z, self.k, dim=-1)
            out = torch.zeros_like(z).scatter_(-1, idx, vals.clamp_min(0))
            return out
        return torch.relu(z)

    def decode(self, z):
        out = z @ self.W_dec
        if self.b_dec is not None:
            out = out + self.b_dec
        return out

    def encode_topk(self, x, k: int) -> tuple[list[int], list[float]]:
        import torch
        with torch.no_grad():
            z = self.encode(x.detach().reshape(1, -1) if x.dim() == 1 else x.detach())
            z = z.reshape(-1, self.d_sae)
            pooled = z.max(dim=0).values
            active = (pooled > 1e-9).nonzero().flatten()
            take = min(k, len(active))
            if take == 0:
                return [], []
            vals, sub = torch.topk(pooled[active], take)
            return active[sub].cpu().tolist(), vals.cpu().tolist()


def load_sae(repo_id: str, hook_layer: int | None = None, hook_component: str | None = None,
             activation: str | None = None, k: int | None = None, revision: str | None = None,
             device: str = "cpu") -> SAE:
    import torch
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id, "config.json", revision=revision)
    with open(cfg_path) as f:
        cfg = json.load(f)
    layer = hook_layer if hook_layer is not None else cfg.get("layer", cfg.get("hook_layer"))
    comp = hook_component if hook_component is not None else cfg.get("component", cfg.get("hook_component", cfg.get("hook_name", "")))
    if layer is None or comp in (None, ""):
        raise ValueError(f"SAE checkpoint {repo_id} does not declare its hook point; pass --sae-layer/--sae-hook explicitly")
    for name in ("weights.pt", "sae.pt", "model.pt"):
        try:
            w_path = hf_hub_download(repo_id, name, revision=revision)
            break
        except Exception:
            w_path = None
    if w_path is None:
        raise FileNotFoundError(f"No weights file found in {repo_id}")
    sd = torch.load(w_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    W_enc = _as_enc(_pick(sd, ["W_enc", "encoder.weight"])).to(device)
    b_enc = _pick(sd, ["b_enc", "encoder.bias"]).to(device)
    W_dec = _pick(sd, ["W_dec", "decoder.weight"]).to(device)
    if W_dec.shape[0] != W_enc.shape[1]:
        W_dec = W_dec.t()
    b_dec = _maybe(sd, ["b_dec", "decoder.bias"])
    if b_dec is not None:
        b_dec = b_dec.to(device)
    d_model, d_sae = W_enc.shape
    act = activation or cfg.get("activation") or "relu"
    kk = k if k is not None else cfg.get("k") or cfg.get("top_k")
    if act == "topk" and kk is None:
        raise ValueError(f"TopK SAE {repo_id} declares no k; pass k explicitly")
    return SAE(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, activation=act,
               k=kk, d_model=d_model, d_sae=d_sae,
               hook_layer=int(layer), hook_component=str(comp), repo_id=repo_id)


def load_sae_local(path: str, hook_layer: int, hook_component: str, activation: str = "relu", k: int | None = None, device: str = "cpu") -> SAE:
    import torch
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    W_enc = _as_enc(_pick(sd, ["W_enc", "encoder.weight"]))
    b_enc = _pick(sd, ["b_enc", "encoder.bias"])
    W_dec = _pick(sd, ["W_dec", "decoder.weight"])
    if W_dec.shape[0] != W_enc.shape[1]:
        W_dec = W_dec.t()
    d_model, d_sae = W_enc.shape
    return SAE(W_enc, b_enc, W_dec, _maybe(sd, ["b_dec"]), activation, k, d_model, d_sae, hook_layer, hook_component, path)


def random_sae(d_model: int = 32, d_sae: int = 256, seed: int = 0, device: str = "cpu") -> SAE:
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    W_enc = torch.randn(d_model, d_sae, generator=g) / (d_model ** 0.5)
    W_dec = W_enc.t().clone()
    W_dec = W_dec / W_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return SAE(W_enc.to(device), torch.zeros(d_sae).to(device), W_dec.to(device), torch.zeros(d_model).to(device),
               "relu", None, d_model, d_sae, -1, "mock")


def validate_sae(activations, sae: SAE) -> dict:
    import torch
    with torch.no_grad():
        x = activations.reshape(-1, sae.d_model).float()
        z = sae.encode(x)
        x_hat = sae.decode(z)
        var = x.var().clamp_min(1e-12)
        fve = float(1 - (x - x_hat).pow(2).mean() / var)
        l0 = float((z > 0).float().sum(dim=-1).mean())
        dead = float(((z > 0).sum(dim=0) == 0).float().mean())
    verdict = "accept" if (fve > 0.7 and 10 <= l0 <= 200 and dead < 0.5) else (
        "reject" if (fve < 0.5 or not (10 <= l0 <= 200) or dead >= 0.5) else "marginal")
    return {"fve": fve, "l0": l0, "dead_frac": dead, "verdict": verdict,
            "hook_layer": sae.hook_layer, "hook_component": sae.hook_component, "repo": sae.repo_id}


def _as_enc(t):
    return t.t() if t.shape[0] > t.shape[1] else t


def _pick(sd: dict, names: list[str]):
    for n in names:
        if n in sd:
            return sd[n]
        for key in sd:
            if key.endswith(n):
                return sd[key]
    raise KeyError(f"None of {names} in checkpoint keys {list(sd)[:20]}")


def _maybe(sd: dict, names: list[str]):
    try:
        return _pick(sd, names)
    except KeyError:
        return None
