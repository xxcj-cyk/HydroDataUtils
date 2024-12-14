import copy
import warnings
import numpy as np
import scipy
import HydroErr as he

def statistic_1d_error(targ_i, pred_i):
    """statistics for one"""
    ind = np.where(np.logical_and(~np.isnan(pred_i), ~np.isnan(targ_i)))[0]
    # Theoretically at least two points for correlation
    if ind.shape[0] > 1:
        xx = pred_i[ind]
        yy = targ_i[ind]
        bias = he.me(xx, yy)
        # RMSE
        rmse = he.rmse(xx, yy)
        # ubRMSE
        pred_mean = np.nanmean(xx)
        target_mean = np.nanmean(yy)
        pred_anom = xx - pred_mean
        target_anom = yy - target_mean
        ubrmse = np.sqrt(np.nanmean((pred_anom - target_anom) ** 2))
        # rho R2 NSE
        corr = he.pearson_r(xx, yy)
        r2 = he.r_squared(xx, yy)
        nse = he.nse(xx, yy)
        kge = he.kge_2009(xx, yy)
        # percent bias
        pbias = np.sum(xx - yy) / np.sum(yy) * 100
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
        pbiaslow = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
        pbiashigh = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
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
        )
    else:
        raise ValueError(
            "The number of data is less than 2, we don't calculate the statistics."
        )

def statistic_nd_error(target: np.array, pred: np.array, fill_nan: str = "no") -> dict:
    """
    Statistics indicators include: Bias, RMSE, ubRMSE, Corr, R2, NSE, KGE, FHV, FLV

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
        Bias, RMSE, ubRMSE, Corr, R2, NSE, KGE, FHV, FLV
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
            # TODO: now all_non_nan_idx is only set for ET, because of its irregular nan values
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
        )
    if fill_nan == "sum":
        for i in range(target.shape[0]):
            tmp = target[i]
            # non_nan_idx = each_non_nan_idx[i]
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
        return out_dict
    elif fill_nan == "mean":
        for i in range(target.shape[0]):
            tmp = target[i]
            # non_nan_idx = each_non_nan_idx[i]
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
    KGE = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
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
                KGE[k] = KGE(xx, yy)
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
    outDict = dict(
        Bias=Bias,
        RMSE=RMSE,
        ubRMSE=ubRMSE,
        Corr=Corr,
        R2=R2,
        NSE=NSE,
        KGE=KGE,
        FHV=PBiashigh,
        FLV=PBiaslow,
    )
    # "The CDF of BFLV will not reach 1.0 because some basins have all zero flow observations for the "
    # "30% low flow interval, the percent bias can be infinite\n"
    # "The number of these cases is " + str(num_lowtarget_zero)
    return outDict



def calculate_and_record_metrics(
    obs, pred, evaluation_metrics, target_col, fill_nan, eval_log
):
    fill_nan_value = fill_nan
    inds = statistic_nd_error(obs, pred, fill_nan_value)

    for evaluation_metric in evaluation_metrics:
        eval_log[f"{evaluation_metric} of {target_col}"] = inds[
            evaluation_metric
        ].tolist()

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