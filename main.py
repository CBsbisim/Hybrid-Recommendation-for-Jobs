# -*- coding: utf-8 -*-
# @Time    : 2024-11-19
# @Author  : chenximin
import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import pymysql
from logger import *
import warnings
warnings.filterwarnings("ignore")

log = Log()
# 判断是否存在csv保存文件的路径，若没有，则创建
if not os.path.exists('data_saved'): os.mkdir('./data_saved')
if not os.path.exists('error_logs'): os.mkdir('./error_logs')

# TOPSIS评分法
def TOPSIS(dataframe):
    for i in range(dataframe.size):
        dataframe[i] = (dataframe.iloc[i]-dataframe.min())/(dataframe.max()-dataframe.min())
    return dataframe

# 用户数据集成,获取contact_count,interview_count，session_count，meaningful_interactions
## 计数有意义的互动数据
def user_log_concat():
    # 获取用户与招聘信息互动数据
    connection=pymysql.connect(host='gz-cdb-qhh48ey7.sql.tencentcdb.com',port=63747,user='algorithm_user',password='guYEdl1NRJyzZnb!',db='bsy_pe_chat',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor() 
    # 获取该数据库下的所有表格名字
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'bsy_pe_chat'"
    table_list = cursor.execute(query)
    table_list = cursor.fetchall()
    # 每个table的提取sql句子
    mapreduce = {'ch_chat_contacts':"SELECT CASE WHEN user_type = 2 THEN user_id WHEN contacts_type = 2 THEN contacts_id END AS job_seeker_id, position_id, COUNT(*) AS contact_count FROM ch_chat_contacts WHERE del_flag = 0 AND position_id IS NOT NULL GROUP BY job_seeker_id, position_id",
                'ch_chat_interview':"SELECT job_hunter_id AS job_seeker_id, position_id, COUNT(*) AS interview_count FROM ch_chat_interview WHERE del_flag = 0 AND position_id IS NOT NULL GROUP BY job_hunter_id, position_id",
                'ch_chat_list':"SELECT l.user_id AS job_seeker_id, c.position_id, COUNT(*) AS session_count, AVG((l.expect_salary_min + l.expect_salary_max) / 2) AS avg_salary FROM ch_chat_list l JOIN ch_chat_contacts c ON l.chat_contacts_id = c.id WHERE l.del_flag = 0 AND l.user_type = 2 AND c.position_id IS NOT NULL GROUP BY l.user_id, c.position_id",
                'ch_chat_log':"SELECT CASE WHEN l.from_type = 2 THEN l.from_id WHEN l.to_type = 2 THEN l.to_id END AS job_seeker_id, c.position_id, COUNT(CASE WHEN l.type IN (3, 4, 16) THEN 1 END) AS meaningful_interactions, MAX(l.create_time) AS last_interaction FROM ch_chat_log l JOIN ch_chat_contacts c ON l.chat_contacts_id = c.id WHERE l.del_flag = 0 AND c.position_id IS NOT NULL GROUP BY job_seeker_id, c.position_id",
                'history_chat_contacts':"SELECT CASE WHEN user_type = 2 THEN user_id WHEN contacts_type = 2 THEN contacts_id END AS job_seeker_id, position_id, COUNT(*) AS contact_count FROM history_chat_contacts WHERE del_flag = 0 AND position_id IS NOT NULL GROUP BY job_seeker_id, position_id",
                'history_chat_interview':"SELECT job_hunter_id AS job_seeker_id, position_id, COUNT(*) AS interview_count FROM history_chat_interview WHERE del_flag = 0 AND position_id IS NOT NULL GROUP BY job_hunter_id, position_id",
                'history_chat_list':"SELECT l.user_id AS job_seeker_id, c.position_id, COUNT(*) AS session_count, AVG((l.expect_salary_min + l.expect_salary_max) / 2) AS avg_salary FROM history_chat_list l JOIN history_chat_contacts c ON l.chat_contacts_id = c.id WHERE l.del_flag = 0 AND l.user_type = 2 AND c.position_id IS NOT NULL GROUP BY l.user_id, c.position_id",
                'history_chat_log':"SELECT CASE WHEN l.from_type = 2 THEN l.from_id WHEN l.to_type = 2 THEN l.to_id END AS job_seeker_id, c.position_id, COUNT(CASE WHEN l.type IN (3, 4, 16) THEN 1 END) AS meaningful_interactions, MAX(l.create_time) AS last_interaction FROM history_chat_log l JOIN history_chat_contacts c ON l.chat_contacts_id = c.id WHERE l.del_flag = 0 AND c.position_id IS NOT NULL GROUP BY job_seeker_id, c.position_id"
                }
    user_log=[]
    # 根据表格类型（chat_contacts，chat_interview，chat_list，chat_log）读取数据，并合并历史数据与当前数据
    for item in table_list[0:4]:
        for jtem in table_list[4:]:
            i=item['TABLE_NAME']
            j=jtem['TABLE_NAME']
            if (i.split('_')[1:]==j.split('_')[1:]) & (i.split('_')[0]!=j.split('_')[0]):
                # 获取数据并合并
                ch = cursor.execute(mapreduce[item['TABLE_NAME']])
                ch = pd.DataFrame(cursor.fetchall())
                history = cursor.execute(mapreduce[jtem['TABLE_NAME']])
                history = pd.DataFrame(cursor.fetchall())
                ch_history = pd.concat([ch,history])
                user_log.append(ch_history)
    # 数据集成，得到每个用户对某一招聘信息的互动数据
    userLog = user_log[0].merge(user_log[1], on=['job_seeker_id', 'position_id'], how='left') \
                        .merge(user_log[2], on=['job_seeker_id', 'position_id'], how='left') \
                        .merge(user_log[3], on=['job_seeker_id', 'position_id'], how='left')
    # 填充缺失值
    userLog = userLog.fillna(0)
    for column in userLog.columns[2:-1]:
        userLog[column] = TOPSIS(userLog[column])
    connection.close()
    print("计数有意义的互动数据....完毕")
    return userLog

# 用户对招聘信息的评分指数
def rating():
    # 1 2 3 4 5
    # Rating=w1*contact_count+w2*interview_count+w3*session_count+w4*meaningful_interactions + 1
    userLog = user_log_concat()
    userLog["rating"] = userLog['avg_salary']
    for i in range(userLog.shape[0]):
        userLog['rating'][i] = (0.5*userLog.iloc[i]['contact_count']+1*userLog.iloc[i]['interview_count']+1.5*userLog.iloc[i]['session_count']+1*userLog.iloc[i]['meaningful_interactions']+1)/5
    ratings = userLog[['job_seeker_id','position_id','rating']]
    ratings.columns = ['user_id','position_id','rating']
    print("用户对招聘信息的评分....完毕")
    return ratings

# 获取数据库的数据
def load_data():    
    connection=pymysql.connect(host='localhost',port=3306,user='root',password='',db='test',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor() 
    query1 = 'SELECT position_id,province,type_id,experience,education,salary_min,salary_max FROM bc_jobs'
    query2 = 'SELECT user_id FROM seeker_user_info'+' WHERE del_flag = 0'
    try:
        # 获取所有招聘数据
        jobs = cursor.execute(query1)
        jobs = pd.DataFrame(cursor.fetchall())
        # 获取所有博思云用户数据
        users = cursor.execute(query2)
        users = pd.DataFrame(cursor.fetchall())
        # 计算用户已浏览招聘数据的评分
        ratings = rating()
    except Exception as ee:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.error('error time:{} \t   error message:{}'.format(now, traceback.format_exc() ))
    finally:
        connection.close()
        print('推荐系统数据加载....完毕')
    return ratings,jobs,users


# 协同过滤+职位相似度过滤的混合推荐器
class HybridRecommender:
    # 基于招聘信息相似度的推荐
    def __init__(self,ratings,jobs,user_id) -> None:
        """
        Args:
            ratings: User ratings DataFrame (user_id, position_id, rating)
            jobs: Job details DataFrame (position_id, job details)
            user_id: Target user ID for recommendations
        """
        # 集合用户对所有招聘信息评分表，包括未评分的招聘信息
        self.ratings = ratings
        self.jobs = jobs
        self.dataset = pd.merge(ratings,jobs, on='position_id', how='left')
        self.user_id = user_id
        # 比例为1:1
        self.alpha = 0.3

    # 基于招聘信息特征的相似度
    def compute_job_similarity(self):
        # 选择清除缺失值
        job_features = self.jobs.dropna()
        # One-hot 编码类别数据
        encoder = OneHotEncoder(sparse_output=False)
        categorical_features = encoder.fit_transform(job_features[['type_id', 'province']])
        # 正则化非类别数据：由于工作经验跟学历的标签值越大越资深，因此两者都为非类型数据
        scaler = MinMaxScaler()
        numerical_features = scaler.fit_transform(job_features[['salary_min', 'salary_max','experience','education']])
        # 合并所有特征
        feature_matrix = pd.concat([pd.DataFrame(categorical_features), pd.DataFrame(numerical_features)], axis=1)
        # 计算职位相似值
        job_similarity = cosine_similarity(feature_matrix)
        return pd.DataFrame(job_similarity, index=job_features['position_id'], columns=job_features['position_id'])
    
    # 混合推荐
    def recommender(self):
        rp = self.dataset.pivot_table(columns=['position_id'],index=['user_id'],values='rating')
        rating_jobs = rp[self.id]
        # 按照用户评分计算招聘信息的相似度
        sim_ratings = rp.corrwith(rating_jobs).fillna(0)
        sim_ratings.columns=['ratings']
        # 按照职位特征计算的相似度
        sim__job_features = self.compute_job_similarity()
        if self.id in sim__job_features.columns:
            sim_features = sim__job_features[self.id]
            sim_features.columns=['ratings']
        else:
            sim_features = sim__job_features[sim__job_features.columns[1]]
        # 混合相似度
        sim_jobs = self.alpha * sim_features + (1 - self.alpha) * sim_ratings
        # 招聘信息对于未查阅的用户评分估算
        rating_c = self.dataset[rating_jobs[self.dataset.user_id].isnull().values & (self.dataset.position_id != self.id)]
        rating_c['similarity'] = rating_c['position_id'].map(sim_jobs.get)
        rating_c['sim_rating'] = rating_c.similarity * rating_c.rating
        recommendation = rating_c.groupby('position_id').apply(lambda s: s.sim_rating.sum() / s.similarity.sum())
        recommendation.columns=['ratings']
        return sim_jobs,recommendation
    
    def get_recommend(self, N):
        """
        Args:
            N: the number of recommendable item   推荐项目数
        Returns:
            top N position_id
        """
        # 按照用户评分的职位进行推荐
        fit_jobs = self.ratings[self.ratings['user_id']==self.user_id]['position_id'].to_list()
        job_recommend=[]
        recommendations = []
        for i in fit_jobs:
            # 按照招聘信息相似度排名，筛选并推荐K个招聘信息
            self.id = i
            sim_jobs, recommendation = self.recommender()
            recommendations.append(recommendation)
        recommendation = pd.concat(recommendations).sort_values(ascending=False)[0:N]
        job_recommend=[i for i in recommendation.index.to_list() if i not in fit_jobs]
        return job_recommend


# 获取当前缓存中最近的csv文件
def get_update_data():
    dir = r'./data_saved'
    file_lists = os.listdir(dir)  #输出文件夹下目录
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn) if not os.path.isdir(dir + "/" + fn) else 0)
    # print(file_lists[-1])
    return 'data_saved/' + file_lists[-1] 

def get_data_for_userid(user_id):
    path = get_update_data()
    new_data = pd.read_csv(path)
    jobs = new_data.loc[new_data['user_id']==user_id,'recommendations'][0][1:-1].replace("'","").split(',')
    return jobs

def process_user(user_id, X_user_ids, position_ids, X, Y):
    """
    Process recommendations for a single user.
    """
    if user_id not in X_user_ids:
        # 如果用户没有互动数据，则随机选择
        recommendations = random.sample(position_ids, 200)
        return user_id, recommendations, 'random'
    else:
        # 如果用户有互动数据，则使用混合推荐器
        current = HybridRecommender(X, Y, user_id)
        recommendations = current.get_recommend(200)
        return user_id, recommendations, 'hybrid'

def save_data_by_cycle():
    X,Y,users= load_data()
    # 需要传入的数据包括： 数据库更新后的数据 更新的周期可自定义，也可以按照默认的时间进行保存
    recommend_for_user =pd.DataFrame(columns=['user_id','recommendations'])
    # 获取id
    user_ids = users['user_id'].to_list()
    recommend_for_user['user_id'] = user_ids
    position_ids = Y['position_id'].to_list()
    X_user_ids = set(X['user_id'].to_list())
    # 针对用户互动情况获取并保存职位推荐
    rand, hybrid = 0, 0
    # 多进程执行
    with ProcessPoolExecutor(max_workers=15) as executor:
        # 提交任务
        futures = {executor.submit(process_user, user_id, X_user_ids, position_ids, X, Y): user_id for user_id in user_ids}
        # 获取推荐结果
        with tqdm(total=len(user_ids), desc="Processing Recommendations", mininterval=0.1) as pbar:
            for future in as_completed(futures):
                try:
                    user_id, recommendations, method = future.result()
                    recommend_for_user.loc[recommend_for_user['user_id'] == user_id, 'recommendations'] = str(recommendations)
                    if method == 'random':
                        rand += 1
                    elif method == 'hybrid':
                        hybrid += 1
                except Exception as e:
                    print(f"Error processing user {futures[future]}: {e}")
                finally:
                    pbar.update(1)
    try:
        save_data = os.path.join('data_saved/','%s.csv' % time.strftime('%Y_%m_%d'))
        recommend_for_user.to_csv(save_data, index=True)
        print("数据保存成功！")
        print("随机推荐用户个数：%d"%rand)
        print("混合推荐用户个数：%d"%hybrid)
    except Exception as ee:
        log.error(traceback.format_exc())
        recommend_for_user.to_csv('data_saved/error.csv', index=False)


if __name__ == '__main__':
    save_data_by_cycle()