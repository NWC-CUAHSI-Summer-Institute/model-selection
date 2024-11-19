# ESSENTIALS
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from datetime import datetime

# CLUSTERING AND RANDOM FOREST
import skfuzzy as fuzz
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# DATA LIBRARIES
import geopandas as gpd
import pandas as pd
import json
import pickle
from xarray.core.dataarray import DataArray

# PREFERENCES
pd.set_option('display.max_columns', 500)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})

# DATA DIRECTORIES
base_dir = r"D:\AAA_Research\CUAHSI_SI_2022\dl\\"
data_dir = f"{base_dir}/data/"
camels_dir = base_dir

hrly_dir = f"{data_dir}hourly_performances/"
HCDN_dir = f"{data_dir}HCDN_nhru_final/"
linear_dir = f"{data_dir}linear_timeseries/"
camelsatt_dir = f"{data_dir}/camels_attributes_v2.0/camels_attributes_v2.0/"
lstmruns_dir = f"{base_dir}full_runs/"
hrlyperf_dir = f"{data_dir}hourly_performances/"
cfeval_dir = f"{base_dir}val_runs/"
nwm_dir = f"{base_dir}NWM_streamflow_results.csv"
stream_dir = f"{base_dir}usgs_streamflow"

class ModelSelector():

    def __init__(self, c_kwargs={}, rf_kwargs={}):
       self.c_kwargs=c_kwargs        # CLUSTERING HYPERPARAMETERS
       self.rf_kwargs=rf_kwargs      # RANDOM FOREST HYPERPARAMETERS
       self.m = 2                    # EXPONENTIATION COEFFICIENT FOR CLUSTERING. TODO: MAKE ADJUSTABLE

    def fuzzyCluster(self, data):
        # Wraps Fuzzy Cluster function, only outputting percent belongs and formal cluster.

        # CHECK THAT REQUIRED FIELDS ARE IN KWARGS, IF NOT ADD
        if "error" not in self.c_kwargs:
            self.c_kwargs['error']=0.005

        if "maxiter" not in self.c_kwargs:
            self.c_kwargs['maxiter']=1000

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, self.n_centers, self.m, **self.c_kwargs)
        label = np.argmax(u, axis=0)
        return cntr, u, fpc, label


    def train_rf(self, X_train, y_train, rf_controls={}):
        # ADAPTED FROM https://stackoverflow.com/questions/28489667/combining-random-forest-models-in-scikit-learn

        # RF CONTROLS PASSED DIRECTLY FROM PARAMETER, DEFAULT IS EMPTY
        rf = RandomForestRegressor(**rf_controls) 

        # RF FITTING 
        rf.fit(X_train, y_train)

        return rf

    def fit(self, attributes, model_perf):

        # CREATE RANDOM FOREST AND TRAIN
        self.rf = self.train_rf(attributes, model_perf, rf_controls=self.rf_kwargs)
        # print(r2_score(self.rf.predict(attributes), model_perf))

        return self

    def predict(self, attributes):

        # CHECK WHETHER MODEL HAS BEEN TRAINED
        if self.rf is None:
            raise(Exception("ModelSelector isn't trained!"))

        # GET RANDOM FOREST PREDICTION
        pred = self.rf.predict(attributes)


        return pred
    


# ADAPTED FROM https://github.com/neuralhydrology/neuralhydrology/blob/1ff36ea8c8eff99ad25fa0f56f0119acbc9e6799/neuralhydrology/evaluation/metrics.py
def nse(obs: DataArray, sim: DataArray) -> float:
    denominator = ((obs - obs.mean())**2).sum()
    numerator = ((sim - obs)**2).sum()
    value = 1 - numerator / denominator
    return float(value)

def getNNSEfromNWM():
    
    # DEFINE DIR WITH ALL STREAMFLOW DATA 
    # /home/ottersloth/data/camels_hourly/
    q_dir = f"{camels_dir}usgs_streamflow"
    
    # TEST PERIOD
    test_start=datetime.strptime("2002-09-30 23:00:00", '%Y-%m-%d %H:%M:%S')
    # ORIGINALLY UNTIL 11 PM, CHANGED BECAUSE COMPARISON FUNCTION IS INCLUSIVE LATER ON
    test_end=datetime.strptime("2007-09-30 22:00:00", '%Y-%m-%d %H:%M:%S') 
    
    # DEFINE AND READ IN NWM TIMESERIES
    nwmdir = f"{cfe_dir}NWM_streamflow_results.csv"
    nwm = pd.read_csv(nwmdir)
    
    # GET BASINS IN DATASET, REMOVE FIRST COLUMN NAME BC IT'S D
    basins = nwm.columns.to_list()[1:]
    
    # CONTAINER FOR OUTPUT
    nnsedf = list()
    
    # LOOP THROUGH BASINS IN DATASET
    for currbasin in basins:
        try:
        
            basinid = currbasin

            # GET MODEL PREDICTION IN CURRENT BASINS
            pred = nwm[currbasin].to_numpy()

            # GET Q FROM USGS FOR TEST PERIOD
            q_read = pd.read_csv(f"{q_dir}/{basinid}-usgs-hourly.csv")
            q_read["datetime"] = pd.to_datetime(q_read['date'], format='%Y-%m-%d %H:%M:%S') # CONVERT TO DATETIME
            q_match = q_read[q_read.datetime.between(test_start, test_end)]
            ts_q = q_match['QObs_CAMELS(mm/h)'].to_numpy()

            # CALCULATE NSE 
            nse_calc = nse(ts_q, pred)

            # CALCULATE NNSE
            nnse = 1 / (2 - nse_calc)

            # APPEND TO CONTAINER
            # df_add = {currbasin: nnse}
            df_add = list(np.array([currbasin, nnse]))
            
            nnsedf.append(df_add)
            print(df_add)
            
        
        except: 
            continue
            
    nnsedf = pd.DataFrame(np.array(nnsedf), columns=["basin_id", "nnse"])
    nnsedf.to_csv(f"{hrly_dir}nwm.csv")
    
    
    return nnsedf

def getFeatureImportance(selector, testidx, input, output, xx, reps=10):
    # USING FULL PERMUTATION IMPORTANCE, AS OUTLINED IN 
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    result = permutation_importance(selector.rf, input.iloc[testidx, :], output.iloc[testidx], n_repeats=reps, random_state=42, n_jobs=2)
    means = pd.Series(result.importances_mean, index=list(input)) 
    std = pd.Series(result.importances_std, index=list(input))
    df = pd.concat([means, std], axis=1)
    df.columns = [str(xx) + "_mean", str(xx) + "_std"]
    return df


def runFitMetric_getMSE(fitmet, rf_kwargs={}):
    # FILEPATH TO SHAPEFILE CONTAINING CAMELS DATASET
    # camelsdir = r"C:\Users\franc\OneDrive - University Of Houston\000_SIResearch\data\HCDN_nhru_final\HCDN_nhru_final_671.shp"
    camelsdir = f"{HCDN_dir}HCDN_nhru_final_671.shp"
    
    # DIRECTORY TO FOLDER CONTAINING CAMELS ATTRIBUTE TEXTFILES
    # PRIOR TO THIS STEP MAKE SURE THE README IN THE FILE SYSTEM HAS BEEN REMOVED (or the file extension has been changed)
    attdir = camelsatt_dir

    # READ CAMELS DATASET
    camels = gpd.read_file(camelsdir)

    # COPY TO KEEP ORIGINAL IN MEMORY
    camels_df = camels 

    # LOOP THROUGH AND JOIN
    filelist = glob.glob(attdir + "*.txt")
    for i, file in enumerate(filelist):
        currdf = pd.read_csv(file, sep=";")
        currdf.rename(columns={'gauge_id': f'gauge_id_{i}'}, inplace=True, errors='raise')
        camels_df = camels_df.merge(currdf, how='left', left_on="hru_id", right_on=f'gauge_id_{i}')

    # DEFINE WHAT WE WANT TO RUN ON
    # perf_dir = r"C:\Users\franc\OneDrive - University Of Houston\000_SIResearch\data\hourly_performances\\"
    perf_dir = hrlyperf_dir
    
    perf_metrics = [fitmet]

    # READ ALL CSV FILES IN DIRECTORY
    os.chdir(perf_dir)
    modelfiles = glob.glob("*.csv")

    # GET FIRST CSV FILE TO DEFINE DATAFRAME AND ADD PREFIX TO COLUMNS BASED ON NAME
    print(perf_dir + modelfiles[0])
    perf = pd.read_csv(perf_dir + modelfiles[0]).add_prefix(modelfiles[0][:-4] + "_")
    # GET COLUMN NAME CONTAINING "BASIN"
    fcol = [col for col in perf.columns if 'basin' in col]

    # LOOP FOR EACH CSV FILE
    for ii in range(1, len(modelfiles)):
        print(perf_dir + modelfiles[ii])
        # GET NEXT CSV FILE TO DEFINE DATAFRAME AND ADD PREFIX TO COLUMNS BASED ON NAME
        currdf = pd.read_csv(perf_dir + modelfiles[ii]).add_prefix(modelfiles[ii][:-4] + "_")

        # GET COLUMN NAME CONTAINING "BASIN"
        basin_col= [col for col in currdf.columns if 'basin' in col]
        # JOIN ON MATCHING BASINS
        perf = perf.merge(currdf, how="inner", left_on=fcol, right_on=basin_col)
    
    # GET COLUMN NAME CONTAINING "FITMET"
    perf_met = [col for col in perf.columns if fitmet in col]

    # CLEAN UP NONSENSICAL DATA (EG, BASIN LABELS)
    # SO LETS GET A LIST OF VARIABLE NAMES WE WANT TO KEEP.

    # TO START WE WILL KEEP THE SAME VARIABLES AS Kratzert et al. 2019, AS SHOWN BY OUR
    # INTERNAL SPREADSHEET Attributes_CAMELS_vs_NHDPlus
    varstokeep = ['organic_frac',
    'elev_mean_x',
    'slope_mean',
    'area_gages2',
    'soil_depth_pelletier',
    'sand_frac',
    'silt_frac',
    'clay_frac',
    'geol_permeability',
    'p_mean',
    'pet_mean',
    'aridity',
    'frac_snow',
    'high_prec_freq',
    'high_prec_dur',
    'low_prec_freq',
    'low_prec_dur']

    camels_df = camels_df.merge(perf, how="inner", left_on="hru_id", right_on=fcol)
    
    inputdataset = camels_df[varstokeep]
    outputdataset = camels_df[perf_met]
    
    
    nsplits = 5
    kf = KFold(n_splits=nsplits, shuffle=True)


    testvalues = np.zeros((inputdataset.shape[0], outputdataset.shape[1]))                  # CONTAINER FOR PERFORMANCE VALUES WHEN BASIN IN TEST SET
    modelno = 0                                                              # COUNTER FOR MODEL CONTAINER
    test_modelno = np.zeros((inputdataset.shape[0], outputdataset.shape[1]))                # MODEL IN WHICH BASIN WAS IN TEST SET
    test_modellist = list()                                                  # MODEL CONTAINER
    featureimportance = list()
    r2train = list()
    r2test = list()
    msetest = list()
    msetrain = list()
    msemeta = list()
    trainlist_x = list()
    trainlist_y = list()

    currout=outputdataset

    # KFOLD SPLIT OF DATASETS
    for train, test in kf.split(inputdataset):            
        # CODE FOR INDIVIDUAL MODEL TRAINING
        for ii in range(currout.shape[1]):
            # TRAIN MODEL ON TRAINING SET
            model = None
            model = ModelSelector(rf_kwargs=rf_kwargs)
            model.fit(inputdataset.iloc[train, :], currout.iloc[train, ii])

            # PERFORM PREDICTION ON TRAIN SET AND GET FIT METRICS
            train_pred = model.predict(inputdataset.iloc[train, :])
            trainrms = mse(train_pred, currout.iloc[train, ii].to_numpy())
            trainr2 = r2_score(train_pred, currout.iloc[train, ii].to_numpy())

            # PERFORM PREDICTION ON TEST SET AND GET FIT METRICS
            model_pred = model.predict(inputdataset.iloc[test, :])
            testrms = mse(model_pred, currout.iloc[test, ii].to_numpy())
            testr2 = r2_score(model_pred, currout.iloc[test, ii].to_numpy())
            
            # GET FEATURE 
            fi = getFeatureImportance(model, test, inputdataset, currout.iloc[:,ii], modelno)
            featureimportance.append(fi)

            # SAVE VALUES IN CONTAINERS ABOVE
            testvalues[test, ii] = model_pred
            test_modelno[test, ii] = modelno
            modelno = modelno + 1
            test_modellist.append(model)
            msetest.append(testrms)
            msetrain.append(trainrms)
            r2train.append(trainr2)
            r2test.append(testr2)
            trainlist_x.append(train_pred)
            trainlist_y.append(outputdataset.iloc[train, :])

            # print(f"Test R2: {testr2:.3f} | Meta R2: {metar2:.3f} | Test MSE: {testrms:.3f} | Meta MSE: {metarms:.3f} | Model ID: {ii}")

    metrics = (r2train, r2test, msetrain, msetest)
    trainlists = (trainlist_x, trainlist_y)
    
    return camels_df, inputdataset, outputdataset, testvalues, test_modelno, test_modellist, featureimportance, metrics, trainlists

def softmax(x):
    # ADAPTED FROM https://www.delftstack.com/howto/numpy/numpy-softmax/
    maxx = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - maxx) #subtracts each row with its max value
    sumx = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sumx 
    return f_x

def ensemblePerf(perf, target, threshold, n=2, softmaxflag = False):
    
    # RANGE OF EACH ROW
    rn = np.max(perf, axis=1, keepdims=True) - perf

    # CALCULATE THE VALUES FOR ENSEMBLING MODELS
    
    # EXPONENTIAL SETUP
    if not softmaxflag:
        perf_w = perf ** n
        perf_w = np.where(rn > threshold, 0, perf_w)
        row_sums = perf_w.sum(axis=1)
        perf_norm = perf_w / row_sums[:, np.newaxis]
    else: 
        perf = np.where(rn > threshold, 0, perf)
        perf_norm = softmax(perf)
    
    n_weighting = np.count_nonzero(np.any(perf_norm == 1, axis=1))
    n_partialweighting = np.count_nonzero(np.any(perf_norm == 0, axis=1))
    n_total = perf_norm.shape[0]
    print(f"Not weighting in {n_weighting} of {n_total} basins, partial weighting in {n_partialweighting - n_weighting}")
    
    return perf_norm



def getLSTMTS(lstm_dir = f'{lstmruns_dir}runs/'):
    file_list = os.listdir(lstm_dir)
    file_list.remove('failed_runs')
    file_list.remove('hydro_signatures')

    num_files = len(file_list)

    lstm_results = {}
    lstm_results['basin_id'] = []
    lstm_results['sim'] = []

    for i in range(num_files): 
        lstm_test_dir = os.path.join(lstm_dir,file_list[i])
        lstm_test_file = os.path.join(lstm_test_dir,"test", "model_epoch003","test_results.p")
        with open(lstm_test_file, "rb") as fp:
            results = pickle.load(fp)

        basin_id = [i for i in results.keys()]

        for j in range(len(basin_id)):
            sim = results[basin_id[j]]['1H']['xr']['QObs_CAMELS(mm/h)_sim'].values
            sim = [float(sim[i]) for i in range(len(sim))]

            lstm_results['sim'].append(sim)
            lstm_results['basin_id'].append(basin_id[j])

    df_lstm_results = pd.DataFrame(lstm_results)
    return df_lstm_results

def getLinearTS(lin_dir = f'{linear_dir}'):
    file_list = os.listdir(lin_dir)

    num_files = len(file_list)

    linear_results = {}
    linear_results['basin_id'] = []
    linear_results['sim'] = []
    
    

    for i in range(num_files): 
        lin_test_dir = os.path.join(lin_dir,file_list[i])
        
        basin_id = file_list[i][4:-4]
        res = pd.read_csv(lin_test_dir+".csv")
        
        linear_results['sim'].append(res["QObs"])
        linear_results['basin_id'].append(basin_id)
        
    df_linear_results = pd.DataFrame(linear_results)
    return df_linear_results

def getWeightedFit(rfargs = {"n_estimators": 15}, weightparams={"threshold":0.05, "n":1, "softmax": False}):
    
    # RUN RF MODELS AND GET OUT OF BAG (OOB) "TESTVALUES"
    camels_df, inputs, target, testvalues, test_modelno, test_modellist, featureimportance, metrics, trainlists = runFitMetric_getMSE("nnse", rf_kwargs=rfargs)
    outstruct_rf = (camels_df, inputs, target, testvalues, test_modelno, test_modellist, featureimportance, metrics, trainlists)
    
    # GET WEIGHTS FROM OOB TESTVALUES
    weights = ensemblePerf(testvalues, target, **weightparams)
    
    # ADD WEIGHTS TO CAMELS AND DEFINE RUNNING DF 
    camels_df['weight_CFE'] = weights[:, 0]
    camels_df['weight_linear'] = weights[:, 1]
    camels_df['weight_LSTM'] = weights[:, 2]
    camels_df['weight_NWM'] = weights[:, 3]
    weights_df = camels_df[['hru_id', 'weight_CFE', 'weight_LSTM', 'weight_NWM', 'weight_linear']]
    
    # GET TIMSERIES OF LSTM RESULTS
    df_lstm_results = getLSTMTS()
    
    # DEFINE DIR WITH ALL STREAMFLOW DATA 
    q_dir = stream_dir

    # TEST PERIOD
    test_start=datetime.strptime("2002-09-30 23:00:00", '%Y-%m-%d %H:%M:%S')
    # ORIGINALLY UNTIL 11 PM, CHANGED BECAUSE COMPARISON FUNCTION IS INCLUSIVE LATER ON
    test_end=datetime.strptime("2007-09-30 22:00:00", '%Y-%m-%d %H:%M:%S') 

    # GET ALL CFE VALIDATION FILES 
    cfe_dir = cfeval_dir
    os.chdir(cfe_dir)
    filelist = glob.glob("*.json")
    
    # GET NWM FILES
    nwmdir = nwm_dir
    nwm = pd.read_csv(nwmdir)

    # GET TIMSERIES OF LINEAR RESULTS
    df_linear_results = getLSTMTS()
    
    basinlist = list()
    outnnselist = list()
    
    for i in range(len(filelist)): 

        # GET FILE NAME FROM CONTAINER
        file = filelist[i]

        # GET BASIN ID BY SPLITTING FILENAME
        basinid_raw = file.split("_")[0]
        basinid = '%08d' % int(basinid_raw) # ZERO PADDING

        # GET TIMESERIES FROM LSTM DATAFRAME
        match = df_lstm_results[df_lstm_results["basin_id"] == basinid]

        # SINCE NOT THE SAME BASINS WERE RUN, CHECK WE ACTUALLY GOT A MATCH
        if match.shape[0] != 1:
            print(f"Skipping {basinid}")
            continue
            
        # CONVERT TIMESERIES TO NP ARRAY
        ts_lstm = np.array(match.iloc[0, 1])
        
        # GET TIMESERIES FROM LINEAR DATAFRAME
        match_2 = df_linear_results[df_linear_results["basin_id"] == basinid]

        # SINCE NOT THE SAME BASINS WERE RUN, CHECK WE ACTUALLY GOT A MATCH
        if match_2.shape[0] != 1:
            print(f"Skipping {basinid}")
            continue
        
        # CONVERT TIMESERIES TO NP ARRAY
        ts_linear = np.array(match_2.iloc[0, 1])
        

        # NOW READ CFE FILE
        with open(file) as json_file:
            data = json.load(json_file)

        # CONVERT TIMESERIES TO NUMPY ARRAY
        ts_cfe = np.array(data['validation sims'])
        
        # GET NWM TIMESERIES
        ts_nwm = nwm[basinid].to_numpy()

        # GET MODEL WEIGHTS 
        weights = weights_df[weights_df["hru_id"] == int(basinid)].to_numpy()
        weights = weights[0][1:]

        # weights = weights[1:]

        # APPLY WEIGHTS
        ts_final = (weights[0] * ts_cfe + weights[1] * ts_linear + ts_lstm  * weights[2] + ts_nwm * weights[3])

        # GET Q FROM USGS FOR TEST PERIOD
        q_read = pd.read_csv(f"{q_dir}/{basinid}-usgs-hourly.csv")
        q_read["datetime"] = pd.to_datetime(q_read['date'], format='%Y-%m-%d %H:%M:%S') # CONVERT TO DATETIME
        q_match = q_read[q_read.datetime.between(test_start, test_end)]
        ts_q = q_match['QObs_CAMELS(mm/h)'].to_numpy()

        # CALCULATE NSE 
        nse_calc = nse(ts_q, ts_final)

        # CALCULATE NNSE
        nnse = 1 / (2 - nse_calc)

        # print(f"{basinid} {nnse}")
        if i % 100 == 0:
            print(f"{i} / {len(filelist)}")

        # SAVE IN PROPER FORMAT
        basinlist.append(basinid)
        outnnselist.append(nnse)
        
    return basinlist, outnnselist, outstruct_rf

def getList(r, ax, c, featureimportance, real_labels, topn=5):

    features = featureimportance[0].shape[0] # NUMBER OF FEATURES
    sum = np.zeros((features, ))
    ss = np.zeros((features, ))
    for i in r:
        # POOL MEANS AND STANDARD DEVIATIONS:
        # - GROUP MEAN IS MEAN OF ALL MEANS 
        # - GROUP STD IS DEFINED BY SQRT(SUM(STDs^2)), SINCE ALL SAMPLE SIZES ARE EQUAL
        currdf = featureimportance[i]
        sum = sum + np.abs(currdf[str(i) + "_mean"].to_numpy()).T
        ss =  ss + currdf[str(i) + "_std"].to_numpy().T

    means = sum / len(featureimportance)
    means = means/means.sum() * 100 # NORMALIZE TO PERCENT
    
    std = np.sqrt(ss)  
    
    ind = np.argsort(means)[-topn:]
    
    # labels = featureimportance[0].index.values[ind]
    labels = real_labels[ind]
    selmeans = means[ind]
    
    # REVERSE TO SHOW IN DESCENDING ORDER
    #selmeans = selmeans[::-1]
    #labels = labels[::-1]
    
    ax.barh(np.arange(selmeans.shape[0]), selmeans, color=c)
    ax.set_yticks(np.arange(selmeans.shape[0]))
    ax.set_yticklabels(labels, fontsize=24)
    ax.set_xlim((0, 90))
    ax.set_xlabel("Perc. Variable Importance")
    
    return ax, means, featureimportance[0].index.values

def prettify(ax, title, string):
    props = dict(facecolor='white', alpha=0.5)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("Prediction of NNSE", fontsize=24)
    ax.set_ylabel("NNSE Value", fontsize=24)
    #lims = [
    #    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    #]
    lims = [0,  # min of both axes
        0.8,  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=.5, zorder=0)
    ax.text(0.05, 0.95, string, transform=ax.transAxes, fontsize=24, verticalalignment='top', bbox=props, zorder=0)
    
    
    ax.grid()
    return ax