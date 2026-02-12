# drct/models/drct_kd_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

try:
    from basicsr.archs import build_network
except Exception:
    build_network = None


@MODEL_REGISTRY.register()
class DRCTKDModel(SRModel):
    """
    Online KD (KS-style)
      L_total   = alpha * L_student + (1-alpha) * L_KD
      L_student = |GT - S_out|_1
      L_KD      = w_out*|T_out - S_out|_1 + w_feat*sum_i MSE(S_feat_i_proj, T_feat_i)

    - student: self.net_g (SRModel이 opt['network_g']로 자동 생성)
    - teacher: self.net_t (여기서 opt['network_t']로 생성 + pth 로드)
    """

    def __init__(self, opt):
        super().__init__(opt)

        assert build_network is not None, "basicsr.archs.build_network import 실패. BasicSR 설치/버전 확인 필요"
        assert "network_t" in opt, "opt에 network_t(teacher 설정)가 필요합니다."

        # -------------------------
        # 1) Teacher build
        # -------------------------
        self.net_t = build_network(opt["network_t"]).to(self.device)
        self.net_t.eval()
        for p in self.net_t.parameters():
            p.requires_grad = False

        # -------------------------
        # 2) Teacher weight load
        # -------------------------
        t_path = opt["path"].get("pretrain_network_t", None)
        if t_path:
            param_key_t = opt["path"].get("param_key_t", "params_ema")
            strict_t = opt["path"].get("strict_load_t", True)

            state = torch.load(t_path, map_location="cpu")
            if isinstance(state, dict) and param_key_t in state:
                state = state[param_key_t]
            self.net_t.load_state_dict(state, strict=strict_t)

        # -------------------------
        # 3) KD loss 설정
        # -------------------------
        kd_opt = opt["train"].get("kd_opt", {})
        self.alpha = float(kd_opt.get("alpha", 0.5))  # student loss 비중
        self.w_out = float(kd_opt.get("w_out", 1.0))
        self.w_feat = float(kd_opt.get("w_feat", 1.0))

        self.l1 = nn.L1Loss(reduction="mean")
        self.l2 = nn.MSELoss(reduction="mean")

        # feat KD에서 채널이 다를 때 맞추기 위한 1x1 projection들(학습됨)
        # key: "cs_ct" 형태
        self.feat_proj = nn.ModuleDict()

    @staticmethod
    def _select_teacher_feats(feat_t, target_len: int):
        """teacher feature list를 student 길이에 맞춰 균등 샘플링."""
        if target_len <= 0:
            return []
        if len(feat_t) == target_len:
            return list(feat_t)
        if target_len == 1:
            return [feat_t[-1]]
        idx = [round(i * (len(feat_t) - 1) / (target_len - 1)) for i in range(target_len)]
        return [feat_t[i] for i in idx]

    def _get_or_make_proj(self, cs: int, ct: int) -> nn.Module:
        """
        student 채널(cs) -> teacher 채널(ct)로 맞추는 1x1 conv.
        새로 만들면 optimizer_g에 param_group으로 추가해서 학습되게 함.
        """
        key = f"{cs}_{ct}"
        if key not in self.feat_proj:
            proj = nn.Conv2d(cs, ct, kernel_size=1, bias=False).to(self.device)
            # 아주 작은 초기화(안정)
            nn.init.normal_(proj.weight, mean=0.0, std=1e-3)
            self.feat_proj[key] = proj

            # optimizer에 projection 파라미터 추가(중요!)
            if hasattr(self, "optimizer_g") and self.optimizer_g is not None:
                self.optimizer_g.add_param_group({"params": proj.parameters()})
        return self.feat_proj[key]

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # -------------------------
        # Student forward
        # -------------------------
        out_s, feat_s = self.net_g(self.lq, return_feat=True)

        # -------------------------
        # Teacher forward (no grad)
        # -------------------------
        with torch.no_grad():
            out_t, feat_t = self.net_t(self.lq, return_feat=True)

        # -------------------------
        # L_student
        # -------------------------
        l_student = self.l1(out_s, self.gt)

        # -------------------------
        # KD output loss
        # -------------------------
        l_kd_out = self.l1(out_s, out_t)

        # -------------------------
        # KD feature loss (항상 tensor로)
        # -------------------------
        l_kd_feat = torch.zeros((), device=self.device)
        kd_feat_count = 0

        if isinstance(feat_s, (list, tuple)) and isinstance(feat_t, (list, tuple)):
            feat_t_use = self._select_teacher_feats(feat_t, target_len=len(feat_s))

            for fs, ft in zip(feat_s, feat_t_use):
                if not (torch.is_tensor(fs) and torch.is_tensor(ft)):
                    continue

                ft = ft.detach()

                # (B,C,H,W) 형태만 처리
                if fs.dim() != 4 or ft.dim() != 4:
                    continue

                # spatial mismatch -> teacher를 student 크기로
                if fs.shape[2:] != ft.shape[2:]:
                    ft = F.interpolate(ft, size=fs.shape[2:], mode="bilinear", align_corners=False)

                # channel mismatch -> student에 1x1 proj 적용
                cs, ct = fs.shape[1], ft.shape[1]
                if cs != ct:
                    proj = self._get_or_make_proj(cs, ct)
                    fs_use = proj(fs)
                else:
                    fs_use = fs

                l_kd_feat = l_kd_feat + self.l2(fs_use, ft)
                kd_feat_count += 1

        # -------------------------
        # total loss
        # -------------------------
        l_kd = self.w_out * l_kd_out + self.w_feat * l_kd_feat
        loss = self.alpha * l_student + (1.0 - self.alpha) * l_kd

        loss.backward()
        self.optimizer_g.step()

        # EMA (SRModel에 있으면 업데이트)
        if hasattr(self, "net_g_ema") and self.net_g_ema is not None:
            decay = self.opt["train"].get("ema_decay", 0)
            if decay and decay > 0:
                self.model_ema(decay=decay)

        self.log_dict = {
            "l_student": l_student.detach(),
            "l_kd_out": l_kd_out.detach(),
            "l_kd_feat": l_kd_feat.detach(),
            "l_kd": l_kd.detach(),
            "l_total": loss.detach(),
            "kd_feat_count": torch.tensor(float(kd_feat_count), device=self.device),
        }
        self.output = out_s
