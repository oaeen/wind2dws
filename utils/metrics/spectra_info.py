import numpy as np
from loguru import logger


def generate_freq(new_n=None, spec_source_type=None):
    """
    ori_n: 原始频率个数
    new_n: 新的频率个数
    n个 0.0345* 1.1^n 等倍频率 ERA5
    [0.0345     0.03795    0.041745   0.0459195  0.05051145 0.0555626
     0.06111885 0.06723074 0.07395381 0.0813492  0.08948411 0.09843253
     0.10827578 0.11910336 0.13101369 0.14411506 0.15852657 0.17437922
     0.19181715 0.21099886 0.23209875 0.25530862 0.28083949 0.30892343
     0.33981578 0.37379736 0.41117709 0.4522948  0.49752428 0.54727671]

    36个 0.0339 * 1.1^n  ifremer
    [0.0339     0.03729    0.041019   0.0451209  0.04963299 0.05459629
     0.06005592 0.06606151 0.07266766 0.07993443 0.08792787 0.09672066
     0.10639272 0.11703199 0.12873519 0.14160871 0.15576958 0.17134654
     0.1884812  0.20732932 0.22806225 0.25086847 0.27595532 0.30355085
     0.33390594 0.36729653 0.40402618 0.4444288  0.48887168 0.53775885
     0.59153474 0.65068821 0.71575703 0.78733273 0.86606601 0.95267261]
    """
    freq = None
    base = None
    ori_n = None

    if spec_source_type is None:
        logger.error("spec_source_type is None, Nothing has been done")
    if spec_source_type == "CDIP":
        freq = get_cdip_buyo_freq()
    if spec_source_type == "IOWAGA":
        ori_n = 36
        base = 0.0339
    if spec_source_type == "ERA5":
        ori_n = 30
        base = 0.0345

    ratio = 1.1
    new_n = ori_n if new_n is None else new_n
    freq = base * ratio ** np.linspace(0, ori_n - 1, new_n)
    return freq


def get_cdip_buyo_freq():
    cdip_freq = [
        0.025,
        0.03,
        0.035,
        0.04,
        0.045,
        0.05,
        0.055,
        0.06,
        0.065,
        0.07,
        0.075,
        0.08,
        0.085,
        0.09,
        0.095,
        0.10125,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.27,
        0.28,
        0.29,
        0.3,
        0.31,
        0.32,
        0.33,
        0.34,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.41,
        0.42,
        0.43,
        0.44,
        0.45,
        0.46,
        0.47,
        0.48,
        0.49,
        0.5,
        0.51,
        0.52,
        0.53,
        0.54,
        0.55,
        0.56,
        0.57,
        0.58,
    ]

    return np.asarray(cdip_freq)


def generate_extend_dirs(n=25, start=0, spec_source_type=None):
    """
    产生 25/73 个等距角度并转换为弧度, 用于插值/绘图, 避免使用24个方向导致图形将在初始角度处出现断裂
    使图形看起来不太美观
    [  0.  15.  30.  45.  60.  75.  90. 105.
     120. 135. 150. 165. 180. 195. 210. 225.
     240. 255. 270. 285. 300. 315. 330. 345. 360.]
     [  0.   5.  10.  15.  20.  25.  30.  35.  40.  45.  50.  55.
       60.  65.  70.  75.  80.  85.  90.  95. 100. 105. 110. 115.
      120. 125. 130. 135. 140. 145. 150. 155. 160. 165. 170. 175.
      180. 185. 190. 195. 200. 205. 210. 215. 220. 225. 230. 235.
      240. 245. 250. 255. 260. 265. 270. 275. 280. 285. 290. 295.
      300. 305. 310. 315. 320. 325. 330. 335. 340. 345. 350. 355. 360.]
    [  7.5  22.5  37.5  52.5   67.5  82.5  97.5 112.5
     127.5 142.5 157.5 172.5  187.5 202.5 217.5 232.5
     247.5 262.5 277.5 292.5 307.5 322.5  337.5 352.5 367.5]
    """
    if spec_source_type == "ERA5":
        start = 7.5

    return np.radians(np.linspace(start, start + 360, n, endpoint=True))


def generate_dirs(n=24, start=0, spec_source_type=None):
    """
    产生 24 个等距角度并转换为弧度, 用于积分等操作
    [  0.  15.  30.  45.  60.  75.  90. 105.
     120. 135. 150. 165. 180. 195. 210. 225.
     240. 255. 270. 285. 300. 315. 330. 345.]
    [  0.   5.  10.  15.  20.  25.  30.  35.  40.  45.  50.  55.
      60.  65.  70.  75.  80.  85.  90.  95. 100. 105. 110. 115.
     120. 125. 130. 135. 140. 145. 150. 155. 160. 165. 170. 175.
     180. 185. 190. 195. 200. 205. 210. 215. 220. 225. 230. 235.
     240. 245. 250. 255. 260. 265. 270. 275. 280. 285. 290. 295.
     300. 305. 310. 315. 320. 325. 330. 335. 340. 345. 350. 355.]
    """
    if spec_source_type == "ERA5":
        start = 7.5
    return np.radians(np.linspace(start, start + 360, n, endpoint=False))


def correct_spec_direction(spec_list, source_type=None):
    _, freq, directions = spec_list.shape
    if source_type is None:
        logger.error("source_type is None, Nothing has been done")
    if source_type == "CDIP":
        logger.warning(f"CDIP correct: 对CDIP海浪谱方向进行调整: 由来向调整为去向")

        spec_list = np.roll(spec_list, directions // 2, axis=2)
    if source_type == "IOWAGA":
        logger.warning(
            f"IOWAGA correct: 对海浪谱值进行旋转和逆序, 从数学坐标系还原为罗盘坐标系"
        )
        degree_per_direction = 360 / directions
        rot_deg = -(int(90 / degree_per_direction) + 1)

        spec_list = np.roll(spec_list, rot_deg, axis=2)
        spec_list = np.flip(spec_list, axis=2)
    if source_type == "ERA5":
        logger.warning(f"ERA5 correct: None")

    return spec_list


def get_spec_desc(max_value=20, vmax=1000):
    spec_desc = {
        "xlabel_text": "True wave spectra value (m$^{2}$s)",
        "ylabel_text": "Predicted wave spectra value (m$^{2}$s)",
        "data_type": "spec",
        "unit_text": "m$^{2}$s",
        "max_value": max_value,
        "vmax": vmax,
    }
    return spec_desc


def get_swh_desc(max_value=5, vmax=100):
    swh_desc = {
        "xlabel_text": "True SWH value (m)",
        "ylabel_text": "Predicted SWH value (m)",
        "data_type": "swh",
        "unit_text": "m",
        "max_value": max_value,
        "vmax": vmax,
    }
    return swh_desc


def get_mwp_minus1_desc(max_value=20, vmax=100):
    mwp_minus1_desc = {
        "xlabel_text": "True MWP value (s)",
        "ylabel_text": "Predicted MWP value (s)",
        "data_type": "Tm-1",
        "unit_text": "s",
        "max_value": max_value,
        "vmax": vmax,
    }
    return mwp_minus1_desc


def get_mwp1_desc(max_value=20, vmax=100):
    mwp1_desc = {
        "xlabel_text": "True MWP value (s)",
        "ylabel_text": "Predicted MWP value (s)",
        "data_type": "Tm1",
        "unit_text": "s",
        "max_value": max_value,
        "vmax": vmax,
    }
    return mwp1_desc


def get_mwp2_desc(max_value=16, vmax=100):
    mwp2_desc = {
        "xlabel_text": "True MWP value (s)",
        "ylabel_text": "Predicted MWP value (s)",
        "data_type": "Tm2",
        "unit_text": "s",
        "max_value": max_value,
        "vmax": vmax,
    }
    return mwp2_desc


def get_mwd_desc(max_value=400, vmax=100):
    mwd_desc = {
        "xlabel_text": "True MWD value (deg)",
        "ylabel_text": "Predicted MWD value (deg)",
        "data_type": "mwd",
        "unit_text": "°",
        "max_value": max_value,
        "vmax": vmax,
    }
    return mwd_desc


# if __name__ == "__main__":
#     print(generate_freq(36, 0.0339, 1.1))
#     print(generate_freq(36, 0.0339, 1.1, new_n=72))
#     print(generate_freq(36, 0.0345, 1.1))
#     print(generate_freq(36, 0.0345, 1.1, cdip_buyo=True))
#     print(generate_freq(36, 0.0345, 1.1, new_n=73))
#     print(generate_freq(36, 0.0345, 1.1, cdip_buyo=True, new_n=73))
#     print(generate_extend_dirs(25))
#     print(generate_extend_dirs(73))
#     print(generate_dirs(24))
#     print(generate_dirs(72))
