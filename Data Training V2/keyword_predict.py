"""關鍵詞辨識推論：輸入一句手語的 .npy，輸出模型有把握的關鍵詞。

先執行 data_train.ipynb 產生 keyword_model/demo_config.json (門檻、詞彙、模型設定)，之後：

命令列 (在 Data Training V2 資料夾執行)：
    python keyword_predict.py 某句.npy [另一句.npy ...]

notebook / 其他程式：
    from keyword_predict import KeywordPredictor
    predictor = KeywordPredictor()
    predictor.predict("某句.npy")        # → [("天氣", 0.83), ("陣雨", 0.71)]，沒把握就是 []
    predictor.predict_with_candidates("某句.npy")
        # → 有把握的詞 [("天氣", 0.83)]，其他可能 [("氣象", 0.32), ("北部", 0.28), ...]，兩者合計最多 SHOW_WORDS 個

輸入可以是 Phase5 切出來的 156 維 (03segments) 或 292 維 (03segments_face) .npy；
模型只用手和 pose (臉部特徵在訓練時就遮掉了)，所以兩種都可以。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 跟這支程式放在同一個資料夾的 keyword_model/demo_config.json (data_train.ipynb 第 5 步產生)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "keyword_model" / "demo_config.json"

# 特徵位置 (跟訓練用的 notebook 相同)
LEFT, RIGHT, POSE = slice(0, 63), slice(63, 126), slice(128, 155)
LEFT_FLAG, RIGHT_FLAG, POSE_FLAG, FACE_FLAG = 126, 127, 155, 291
FACE = slice(156, 291)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class SignKeywordCTC(nn.Module):
    """跟 data_train.ipynb 訓練的模型相同。"""

    def __init__(self, n_keywords, feat_dim, d_model, n_heads, enc_layers, ff_dim, dropout):
        super().__init__()
        self.src_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Dropout(dropout), nn.Linear(2 * d_model, n_keywords))
        self.ctc_head = nn.Linear(d_model, n_keywords + 1)

    def forward(self, src, src_mask):
        h = self.encoder(self.pos(self.src_proj(src)), src_key_padding_mask=src_mask)
        valid = (~src_mask).unsqueeze(-1).float()
        mean = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        w = self.attn(h).squeeze(-1).float().masked_fill(src_mask, float("-inf")).softmax(-1)
        attended = (w.unsqueeze(-1) * h.float()).sum(1)
        logits = self.head(torch.cat([mean.float(), attended], dim=-1))
        return logits, self.ctc_head(h).float()


class KeywordPredictor:
    def __init__(self, config_path=DEFAULT_CONFIG, device=None):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"找不到 {config_path}：請先執行 data_train.ipynb 產生設定檔")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        self.cfg = cfg
        self.keywords = cfg["keywords"]
        self.threshold = cfg["threshold"]
        self.max_words = cfg["max_words"]
        self.show_words = cfg.get("show_words", 5)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        norm = np.load(base / cfg["norm_file"])
        self.mean, self.std = norm["mean"], norm["std"]
        m = cfg["model"]

        def load(n_outputs, checkpoint):
            model = SignKeywordCTC(n_outputs, cfg["feat_dim"], m["d_model"], m["n_heads"],
                                   m["enc_layers"], m["ff_dim"], m["dropout"]).to(self.device)
            model.load_state_dict(torch.load(base / checkpoint, map_location=self.device))
            return model.eval()

        self.model = load(len(self.keywords), cfg["checkpoint"])
        # 集成：第一階段模型 (候選詞較多) 的分類層分數，取出最終詞彙對應的欄位，跟第二階段平均
        ens = cfg.get("ensemble")
        self.stage1 = load(ens["n_keywords"], ens["checkpoint"]) if ens else None
        self.stage1_columns = torch.tensor(ens["columns"], device=self.device) if ens else None

    # ------------------------------------------------------------ 前處理 (跟訓練時的驗證 / 測試相同，不做資料增強)
    def valid_mask(self, arr):
        m = np.ones(arr.shape, dtype=bool)
        m[:, LEFT] = arr[:, [LEFT_FLAG]] > 0.5
        m[:, RIGHT] = arr[:, [RIGHT_FLAG]] > 0.5
        point_ok = (arr[:, POSE].reshape(-1, 9, 3)[:, :, 2] > 0) & (arr[:, [POSE_FLAG]] > 0.5)
        m[:, POSE] = np.repeat(point_ok, 3, axis=1)
        m[:, FACE] = arr[:, [FACE_FLAG]] > 0.5
        if not self.cfg["use_face"]:
            m[:, FACE] = False
            m[:, FACE_FLAG] = False
        return m

    def preprocess(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] not in (156, self.cfg["feat_dim"]):
            raise ValueError(f"輸入應該是 (格數, 156) 或 (格數, {self.cfg['feat_dim']})，收到 {arr.shape}")
        if arr.shape[1] == 156:                          # 沒有臉部特徵 → 補 0 (等於「沒偵測到臉」)
            if self.cfg["use_face"]:
                raise ValueError("這個模型有用臉部特徵，輸入需要 292 維 (03segments_face)")
            arr = np.concatenate([arr, np.zeros((len(arr), self.cfg["feat_dim"] - 156), np.float32)], axis=1)
        raw = arr[::self.cfg["frame_stride"]]
        x = np.where(self.valid_mask(raw), (raw - self.mean) / self.std, 0.0).astype(np.float32)
        return x[:self.cfg["max_frames"]]

    # ------------------------------------------------------------ 推論
    @torch.no_grad()
    def scores(self, npy):
        """每個關鍵詞的分數。npy 可以是路徑或陣列。
        第二階段 = 分類層機率與 CTC 偵測分數的平均；有集成設定時，再跟第一階段的分類層機率平均。"""
        arr = np.load(npy) if isinstance(npy, (str, Path)) else npy
        x = torch.from_numpy(self.preprocess(arr)).unsqueeze(0).to(self.device)
        mask = torch.zeros(1, x.size(1), dtype=torch.bool, device=self.device)
        logits, ctc_logits = self.model(x, mask)
        cls = torch.sigmoid(logits.float())[0]
        det = ctc_logits.softmax(-1)[0, :, 1:].amax(0)
        score = (cls + det) / 2
        if self.stage1 is not None:
            stage1_logits, _ = self.stage1(x, mask)
            score = (score + torch.sigmoid(stage1_logits.float())[0, self.stage1_columns]) / 2
        return score.cpu().numpy()

    def predict(self, npy, threshold=None, max_words=None):
        """回傳 [(詞, 分數), ...]，依分數由高到低；沒有任何詞超過門檻就回傳 []。"""
        threshold = self.threshold if threshold is None else threshold
        max_words = self.max_words if max_words is None else max_words
        s = self.scores(npy)
        order = [j for j in np.argsort(-s) if s[j] >= threshold][:max_words]
        return [(self.keywords[j], float(s[j])) for j in order]

    def predict_with_candidates(self, npy, threshold=None, show_words=None):
        """兩層輸出：(有把握的詞, 其他可能的詞)。
        有把握 = 分數 >= 門檻 (準確率約為 data_train.ipynb 第 5 步報告的「有把握」精確率)；
        其他可能 = 沒超過門檻、但分數最高的幾個詞，補到合計 show_words 個 (準確率低很多，只當參考)。"""
        threshold = self.threshold if threshold is None else threshold
        show_words = self.show_words if show_words is None else show_words
        s = self.scores(npy)
        order = list(np.argsort(-s))
        confident = [j for j in order if s[j] >= threshold][:min(self.max_words, show_words)]
        others = [j for j in order if j not in confident][:max(0, show_words - len(confident))]
        return ([(self.keywords[j], float(s[j])) for j in confident],
                [(self.keywords[j], float(s[j])) for j in others])


def main(paths):
    if not paths:
        print(__doc__)
        return
    predictor = KeywordPredictor()
    print(f"門檻 {predictor.threshold:.2f}，每句顯示 {predictor.show_words} 個詞")
    for path in paths:
        confident, others = predictor.predict_with_candidates(path)
        fmt = lambda words: "、".join(f"{w} ({p:.2f})" for w, p in words)
        print(f"{path}")
        print(f"  有把握：{fmt(confident) if confident else '無'}")
        print(f"  其他可能：{fmt(others) if others else '無'}")


if __name__ == "__main__":
    main(sys.argv[1:])
