import copy
import warnings
import numpy as np
import scipy
import HydroErr as he


def pfe(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    mask = ~np.isnan(obs)
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) == 0:
        return np.nan
    peak_obs = np.max(obs)
    peak_sim = np.max(sim)
    if peak_obs == 0:
        return np.inf if peak_sim != 0 else 0
    return (peak_sim - peak_obs) / peak_obs * 100


def pte(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    mask = ~np.isnan(obs)
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) == 0:
        return np.nan
    peak_time_obs = np.argmax(obs)
    peak_time_sim = np.argmax(sim)
    return peak_time_sim - peak_time_obs


def high_rmse(obs, sim, high_ratio=0.8):
    """
    计算高值区域（>= 观测峰值 * high_ratio）的RMSE
    
    Parameters
    ----------
    obs : array-like
        观测值
    sim : array-like
        模拟值
    high_ratio : float
        高值区域阈值比例（相对于观测峰值），默认 0.8
        表示将所有 >= obs_peak * high_ratio 的时段视为"高值区域"
    
    Returns
    -------
    float
        高值区域的RMSE，如果没有高值区域数据则返回 np.nan
    """
    obs = np.array(obs)
    sim = np.array(sim)
    # 同时过滤 obs 和 sim 中的 NaN 和 inf 值，避免数值不稳定
    mask = np.logical_and(
        np.logical_and(~np.isnan(obs), ~np.isnan(sim)),
        np.logical_and(np.isfinite(obs), np.isfinite(sim))
    )
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) == 0:
        return np.nan
    
    # 计算观测峰值
    obs_peak = np.max(obs)
    
    # 如果峰值 <= 0 或为 inf，则高值区域退化为整个数据集
    if obs_peak <= 0 or not np.isfinite(obs_peak):
        high_mask = np.ones_like(obs, dtype=bool)
    else:
        threshold = obs_peak * high_ratio
        high_mask = obs >= threshold
    
    # 筛选高值区域的数据
    if np.sum(high_mask) > 0:
        obs_high = obs[high_mask]
        sim_high = sim[high_mask]
        # 再次检查筛选后的数据是否有效
        if len(obs_high) == 0:
            return np.nan
        # 计算RMSE，使用 np.nanmean 增加健壮性
        diff_sq = (sim_high - obs_high) ** 2
        mean_diff_sq = np.nanmean(diff_sq)
        if not np.isfinite(mean_diff_sq) or mean_diff_sq < 0:
            return np.nan
        rmse_high = np.sqrt(mean_diff_sq)
        # 确保返回值是有限的
        return rmse_high if np.isfinite(rmse_high) else np.nan
    else:
        return np.nan


def statistic_1d_error(targ_i, pred_i):
    """statistics for one"""
    ind = np.where(np.logical_and(~np.isnan(pred_i), ~np.isnan(targ_i)))[0]
    # Theoretically at least two points for correlation
    if ind.shape[0] > 1:
        xx = pred_i[ind]
        yy = targ_i[ind]
        bias = he.me(xx, yy)
        rmse = he.rmse(xx, yy)
        pred_mean = np.nanmean(xx)
        target_mean = np.nanmean(yy)
        pred_anom = xx - pred_mean
        target_anom = yy - target_mean
        ubrmse = np.sqrt(np.nanmean((pred_anom - target_anom) ** 2))
        corr = he.pearson_r(xx, yy)
        r2 = he.r_squared(xx, yy)
        nse = he.nse(xx, yy)
        kge = he.kge_2009(xx, yy)
        pbias = np.sum(xx - yy) / np.sum(yy) * 100
        pred_sort = np.sort(xx)
        target_sort = np.sort(yy)
        indexlow = round(0.3 * len(pred_sort))
        indexhigh = round(0.98 * len(pred_sort))
        lowpred = pred_sort[:indexlow]
        highpred = pred_sort[indexhigh:]
        lowtarget = target_sort[:indexlow]
        hightarget = target_sort[indexhigh:]
        pbiaslow = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
        pbiashigh = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
        pfe_val = pfe(yy, xx)
        pte_val = pte(yy, xx)
        high_rmse_val = high_rmse(yy, xx)
        return dict(
            Bias=bias,
            RMSE=rmse,
            ubRMSE=ubrmse,
            Corr=corr,
            R2=r2,
            NSE=nse,
            KGE=kge,
            FHV=pbiashigh,
            FLV=pbiaslow,
            PFE=pfe_val,
            PTE=pte_val,
            HighRMSE=high_rmse_val,
        )
    else:
        raise ValueError(
            "The number of data is less than 2, we don't calculate the statistics."
        )


def KGE(xs, xo):
    """
    Kling Gupta Efficiency (Gupta et al., 2009, http://dx.doi.org/10.1016/j.jhydrol.2009.08.003)
    input:
        xs: simulated
        xo: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    r = np.corrcoef(xo, xs)[0, 1]
    alpha = np.std(xs) / np.std(xo)
    beta = np.mean(xs) / np.mean(xo)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def statistic_nd_error(target: np.array, pred: np.array, fill_nan: str = "no") -> dict:
    """
    Statistics indicators include: Bias, RMSE, ubRMSE, Corr, R2, NSE, KGE, FHV, FLV, PFE, PTE, HighRMSE

    Parameters
    ----------
    target
        observations, typically 2-dim, when it is 3-dim, set a loop for final dim
    pred
        predictions
    fill_nan
        "no" means ignoring the NaN value, and it is the default setting;
        "sum" means calculate the sum of the following values in the NaN locations.
        For example, observations are [1, nan, nan, 2], and predictions are [0.3, 0.3, 0.3, 1.5].
        Then, "no" means [1, 2] v.s. [0.3, 1.5] while "sum" means [1, 2] v.s. [0.3 + 0.3 + 0.3, 1.5];
        "mean" represents calculate average value the following values in the NaN locations.

    Returns
    -------
    dict
        Bias, RMSE, ubRMSE, Corr, R2, NSE, KGE, FHV, FLV, PFE, PTE, HighRMSE
    """
    if len(target.shape) == 3:
        assert type(fill_nan) in [list, tuple, np.ndarray]
        if type(fill_nan) is not list or len(fill_nan) != target.shape[-1]:
            raise RuntimeError("Please give more fill_nan choices")
    if len(target.shape) == 2 and (type(fill_nan) is list or type(fill_nan) is tuple):
        fill_nan = fill_nan[0]
    assert type(fill_nan) is str
    if fill_nan != "no":
        each_non_nan_idx = []
        all_non_nan_idx = []
        for i in range(target.shape[0]):
            tmp = target[i]
            non_nan_idx_tmp = [j for j in range(tmp.size) if not np.isnan(tmp[j])]
            each_non_nan_idx.append(non_nan_idx_tmp)
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp
            non_nan_idx = np.unique(all_non_nan_idx).tolist()
        # some NaN data appear in different dates in different basins, so we have to calculate the metric for each basin
        # but for ET, it is not very resonable to calculate the metric for each basin in this way, for example,
        # the non_nan_idx: [1, 9, 17, 33, 41], then there are 16 elements in 17 -> 33, so use all_non_nan_idx is better
        # hence we don't use each_non_nan_idx finally
        out_dict = dict(
            Bias=[],
            RMSE=[],
            ubRMSE=[],
            Corr=[],
            R2=[],
            NSE=[],
            KGE=[],
            FHV=[],
            FLV=[],
            PFE=[],
            PTE=[],
            HighRMSE=[],
        )
    if fill_nan == "sum":
        for i in range(target.shape[0]):
            tmp = target[i]
            targ_i = tmp[non_nan_idx]
            pred_i = np.add.reduceat(pred[i], non_nan_idx)
            dict_i = statistic_1d_error(targ_i, pred_i)
            out_dict["Bias"].append(dict_i["Bias"])
            out_dict["RMSE"].append(dict_i["RMSE"])
            out_dict["ubRMSE"].append(dict_i["ubRMSE"])
            out_dict["Corr"].append(dict_i["Corr"])
            out_dict["R2"].append(dict_i["R2"])
            out_dict["NSE"].append(dict_i["NSE"])
            out_dict["KGE"].append(dict_i["KGE"])
            out_dict["FHV"].append(dict_i["FHV"])
            out_dict["FLV"].append(dict_i["FLV"])
            out_dict["PFE"].append(dict_i["PFE"])
            out_dict["PTE"].append(dict_i["PTE"])
            out_dict["HighRMSE"].append(dict_i["HighRMSE"])
        return out_dict
    elif fill_nan == "mean":
        for i in range(target.shape[0]):
            tmp = target[i]
            targ_i = tmp[non_nan_idx]
            pred_i_sum = np.add.reduceat(pred[i], non_nan_idx)
            if non_nan_idx[-1] < len(pred[i]):
                idx4mean = non_nan_idx + [len(pred[i])]
            else:
                idx4mean = copy.copy(non_nan_idx)
            idx_interval = [y - x for x, y in zip(idx4mean, idx4mean[1:])]
            pred_i = pred_i_sum / idx_interval
            dict_i = statistic_1d_error(targ_i, pred_i)
            out_dict["Bias"].append(dict_i["Bias"])
            out_dict["RMSE"].append(dict_i["RMSE"])
            out_dict["ubRMSE"].append(dict_i["ubRMSE"])
            out_dict["Corr"].append(dict_i["Corr"])
            out_dict["R2"].append(dict_i["R2"])
            out_dict["NSE"].append(dict_i["NSE"])
            out_dict["KGE"].append(dict_i["KGE"])
            out_dict["FHV"].append(dict_i["FHV"])
            out_dict["FLV"].append(dict_i["FLV"])
            out_dict["PFE"].append(dict_i["PFE"])
            out_dict["PTE"].append(dict_i["PTE"])
            out_dict["HighRMSE"].append(dict_i["HighRMSE"])
        return out_dict
    # TODO: refactor Dapeng's code
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target) ** 2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    KGe = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    PFE_arr = np.full(ngrid, np.nan)
    PTE_arr = np.full(ngrid, np.nan)
    HighRMSE_arr = np.full(ngrid, np.nan)
    num_lowtarget_zero = 0
    for k in range(ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100
            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                yymean = yy.mean()
                SST = np.sum((yy - yymean) ** 2)
                SSReg = np.sum((xx - yymean) ** 2)
                SSRes = np.sum((yy - xx) ** 2)
                R2[k] = 1 - SSRes / SST
                NSE[k] = 1 - SSRes / SST
                KGe[k] = KGE(xx, yy)
            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            if np.sum(lowtarget) == 0:
                num_lowtarget_zero = num_lowtarget_zero + 1
            with warnings.catch_warnings():
                # Sometimes the lowtarget is all 0, which will cause a warning
                # but I know it is not an error, so I ignore it
                warnings.simplefilter("ignore", category=RuntimeWarning)
                PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
            PFE_arr[k] = pfe(yy, xx)
            PTE_arr[k] = pte(yy, xx)
            HighRMSE_arr[k] = high_rmse(yy, xx)
    outDict = dict(
        Bias=Bias,
        RMSE=RMSE,
        ubRMSE=ubRMSE,
        Corr=Corr,
        R2=R2,
        NSE=NSE,
        KGE=KGe,
        FHV=PBiashigh,
        FLV=PBiaslow,
        PFE=PFE_arr,
        PTE=PTE_arr,
        HighRMSE=HighRMSE_arr,
    )
    return outDict


def calculate_and_record_metrics(
    obs, pred, evaluation_metrics, target_col, fill_nan, eval_log
):
    fill_nan_value = fill_nan
    inds = statistic_nd_error(obs, pred, fill_nan_value)

    # 控制输出到日志中的小数位数（不影响内部计算精度）
    rounding_map = {
        "NSE": 4,
        "KGE": 4,
        "RMSE": 4,
        "HighRMSE": 4,
        "PFE": 3,
        "PTE": 2,
    }

    for evaluation_metric in evaluation_metrics:
        values = inds[evaluation_metric]
        if evaluation_metric in rounding_map:
            values = np.round(values, rounding_map[evaluation_metric])
        eval_log[f"{evaluation_metric} of {target_col}"] = values.tolist()

    return eval_log


def cal_4_stat_inds(b):
    """
    Calculate four statistics indices: percentile 10 and 90, mean value, standard deviation

    Parameters
    ----------
    b
        input data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat(x: np.array) -> list:
    """
    Get statistic values of x (Exclude the NaN values)

    Parameters
    ----------
    x: the array

    Returns
    -------
    list
        [10% quantile, 90% quantile, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]
    if b.size == 0:
        # if b is [], then give it a 0 value
        b = np.array([0])
    return cal_4_stat_inds(b)


def cal_stat_gamma(x):
    """
    Try to transform a time series data to normal distribution

    Now only for daily streamflow, precipitation and evapotranspiration;
    When nan values exist, just ignore them.

    Parameters
    ----------
    x
        time series data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)


def cal_stat_prcp_norm(x, meanprep):
    """
    normalized variable by precipitation with cal_stat_gamma

    dividing a var with prcp means we can get a normalized var without rainfall's magnitude's influence,
    so that we don't have bias for dry and wet basins

    Parameters
    ----------
    x
        data to be normalized
    meanprep
        meanprep = readAttr(gageDict['id'], ['p_mean'])

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    # unit (mm/day)/(mm/day)
    flowua = x / tempprep
    return cal_stat_gamma(flowua)
