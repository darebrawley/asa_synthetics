# %load_ext autoreload
# %autoreload 2
from synthpop.recipes.starter2 import Starter
from synthpop.synthesizer import synthesize_all, synthesize, enable_logging
import os
import pandas as pd
import csv
enable_logging()

# setting API Key
os.environ["CENSUS"] = "d95e144b39e17f929287714b0b8ba9768cecdc9f"


import logging
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import chisquare

logger = logging.getLogger("synthpop")
FitQuality = namedtuple(
    'FitQuality',
    ('people_chisq', 'people_p'))
BlockGroupID = namedtuple(
    'BlockGroupID', ('state', 'county', 'tract', 'block_group'))


def enable_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)



def synthesize_all2(recipe, num_geogs=None, indexes=None,
                   marginal_zero_sub=.01, jd_zero_sub=.001):
    """
    Returns
    -------
    households, people : pandas.DataFrame
    fit_quality : dict of FitQuality
        Keys are geographic IDs, values are namedtuples with attributes
        ``.household_chisq``, ``household_p``, ``people_chisq``,
        and ``people_p``.

    """
    print("Synthesizing at geog level: '{}' (number of geographies is {})"
          .format(recipe.get_geography_name(), recipe.get_num_geographies()))

    if indexes is None:
        indexes = recipe.get_available_geography_ids()

    hh_list = []
    people_list = []
    cnt = 0
    fit_quality = {}
    hh_index_start = 0

    # TODO will parallelization work here?
    for geog_id in indexes:
        print("Synthesizing geog id:\n", geog_id)

        h_marg = recipe.get_household_marginal_for_geography(geog_id)
        logger.debug("Household marginal")
        logger.debug(h_marg)

        p_marg = recipe.get_person_marginal_for_geography(geog_id)
        logger.debug("Person marginal")
        logger.debug(p_marg)

        h_pums, h_jd = recipe.\
            get_household_joint_dist_for_geography(geog_id)
        logger.debug("Household joint distribution")
        logger.debug(h_jd)

        p_pums, p_jd = recipe.get_person_joint_dist_for_geography(geog_id)
        logger.debug("Person joint distribution")
        logger.debug(p_jd)

        households, people, people_chisq, people_p = \
            synthesize(
                h_marg, p_marg, h_jd, p_jd, h_pums, p_pums,
                marginal_zero_sub=marginal_zero_sub, jd_zero_sub=jd_zero_sub,
                hh_index_start=hh_index_start)

        # Append location identifiers to the synthesized households
        for geog_cat in geog_id.keys():
            households[geog_cat] = geog_id[geog_cat]
        for geog_cat in geog_id.keys():
            people[geog_cat] = geog_id[geog_cat]


        hh_list.append(households)
        people_list.append(people)
        key = BlockGroupID(
            geog_id['state'], geog_id['county'], geog_id['tract'],
            geog_id['block group'])
        fit_quality[key] = FitQuality(people_chisq, people_p)

        cnt += 1
        if len(households) > 0:
            hh_index_start = households.index.values[-1] + 1

        if num_geogs is not None and cnt >= num_geogs:
            break

    # TODO might want to write this to disk as we go?
    all_households = pd.concat(hh_list)
    all_persons = pd.concat(people_list)

    return (all_households, all_persons, fit_quality)

# funciton that save a csv for the results for each tract it is given
def synthesize_save(tract):
    print("tract number %s" %tract)
    starter = Starter(os.environ["CENSUS"], "NC", "Mecklenburg County",tract)
    print("tract number %s" %tract)
    all_households, all_persons, fit_quality = synthesize_all2(starter)
    all_households.to_csv('data_outputs/%s_households.csv'%tract)
    all_persons.to_csv('data_outputs/%s_persons.csv'%tract)
    with open('data_outputs/%s_fit.csv'%tract, 'w') as f:
        for key in fit_quality.keys():
            f.write("%s,%s\n"%(key,fit_quality[key]))

    #function that iterates over multiple tracts
def synthesize_tracts_save(tracts):
    for tract in tracts:
        synthesize_save(tract)

tracts1 = ["006209","005521","005511","005614","005611","005200","002003","003105","980300","003007","003013","005815"]
tracts2 = ["000900","000100","001000","000300","001100","001200","001300","005715","005828","005836","003806","005834","003805"]
tracts3 = ["003203","003108","002702","001609","001504","005843","005616","001505","005841","001507","001603","006007","005512","006303","001921","005716"]
tracts4 = ["001922","002906","005830","000400","005832","000500","003204","000600","980200","000700","005912","005909","005906","006008","006215","000800","005508","005509","005518","005522","005714","006006","005826","005847","005842","005908","006211","006210","006304","006404","006403","005618","005917","006212","005915","001923","004303","005517","001917","005524","005520","005835","005617","005604","005838","005845","006104","004000","004100","004200","005615","005612","006407","006108","005825","006103","003802","006405","003902","005717","005848","006009"]
tracts5=["005605","005610","005706","005911","005709","006214","005712","005846","005713","004302","005000","005823","004500","006107","006203","004600","005609","001912","004700","005833","003006","004800","001509","005812","005100","001801","005817","005301","001608","003109","005827","001915","006109","001702","001605","001802"]
tracts6 = ["001607","001606","005305","001701","001920","001910","003017","005840","005910","005510","005513","002002","003016","003102","006213","003103","005837","005514","005619","980100","005307","005306","001508","005613","005916","003808","005918","001510","005914","005308"]
tracts7 = ["005523","001919","001914","001918","002100","001916","002200","002300","002400","005621","005620","002905","002500","005831","003201","003300","003400","005829","003500","005839","003106","003018","005913","003807","005907","003903","006005","002600","002800","003011","003600"]
tracts4rep = ["005842","005908","006211","006210","006304","006404","006403","005618","005917","006212","005915","001923","004303","005517","001917","005524","005520","005835","005617","005604","005838","005845","006104","004000","004100","004200","005615","005612","006407","006108","005825","006103","003802","006405","003902","005717","005848","006009"]
tracts8= ["003700","004305","004304","006010","006208","002004","006406","003008","005711","003012","002903","005811","002904","005710","001911","003015","004400","005401","006204","001400","006302","005824","002701","005404","005403","006106","005519","005844","005516","004900","005515","006105","005816"]
tracts4rep2 = ["005604","005838","005845","006104","004000","004100","004200","005615","005612","006407","006108","005825","006103","003802","006405","003902","005717","005848","006009"]
tracts980200_100 = ["980200","980100"]
synthesize_tracts_save(tracts1)
