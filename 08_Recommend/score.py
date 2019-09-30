# coding: utf-8
import os
import sys
import pandas as pd
import numpy as np

from datetime import datetime, timedelta as delta
import statistics, math

import math # for novelty, isnan
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

def _get_actual_table(recommend_table, score_threshold, test_table, context_count_table, topk=None):
    """
    score_threshold以上のrecommend_tableを抜粋し，レコメンドした時のテスト期間での実績テーブルを返す
    test_tableは，１回以上利用があったレコードしか無いので，そもそも状況が発生したのかの情報を追加するために，テスト期間での状況の実績テーブル（context_count_table)も入れることに注意 
    また、topkは各ユーザ、ジャンルごとに上位kの「時間帯、状況」の組み合わせを抜粋する
    """
    # 利用確度が閾値以上であればレコメンド
    recommend_table_filtered = recommend_table[recommend_table['score'] >= score_threshold]
    
    # 利用確度がtopk以上のものをレコメンド
    if topk is not None:
        recommend_table_filtered = recommend_table_filtered.reset_index(drop=True) # index振り直し
        topk_table = pd.DataFrame()
        
        # ジャンルごとにtopk以上のものをレコメンド
        genre_id_list = recommend_table_filtered['genre_id'].unique()
        for genre_id in genre_id_list:
            recommend_table_genre = recommend_table_filtered[recommend_table_filtered['genre_id']==genre_id]
            # ユーザごとにtopk以上のものをレコメンド
            tmp = recommend_table_genre
            for k in range(topk):
                topk = tmp.sort_values(['cuid', 'score'], ascending=False).drop_duplicates(['cuid'], keep='first')
                topk_table = pd.concat([topk_table, topk])
                tmp = tmp.drop(topk.index)
                
        recommend_table_filtered = topk_table
            
    # レコメンドテーブルは必要なテストデータを抽出する上で必要なカラムだけにする
    #column = ['cuid', 'time', 'poi_type', 'context', 'genre_id']
    column = ['cuid', 'time', 'poi_type', 'genre_id']
    recommend_table_filtered = recommend_table_filtered[column]
    
    # テストテーブルと外部結合（レコメンドテーブルに対する左外部結合）
    actual_table = pd.merge(recommend_table_filtered, test_table, on=column, how='left')
    
    # null（レコメンドテーブルに存在するが，テストテーブルになかったもの）は0で埋める
    actual_table = actual_table.fillna(0)
    
    # このままだと，
    # 「レコメンドテーブルに存在したが，テストテーブルに存在しなかった時間帯/状況の組み合わせ」の「context_num」が０になっていることに注意
    # 実際はその組み合わせは発生しているはずなので実績値で埋める必要あり
    # なぜテストテーブルに最初から作っていなかったかというと，レコメンドテーブルには不必要なデータであるため
    # そこで，検証で使うために，ここでデータを修正する
    # 突合するために少し修正
    actual_table = actual_table.drop('context_num', axis=1)
    
    # context_numが0のものを修正したいだけなので，内部結合
    #################################################################################################
    # レコメンドテーブルによりレコメンドした時間帯、状況がテスト期間で1度も利用がなかった場合もある
    #################################################################################################
    #column = ['cuid', 'time', 'poi_type', 'context']
    column = ['cuid', 'time', 'poi_type']
    actual_table = pd.merge(actual_table, context_count_table, on=column, how='inner')
    
    actual_delivery_user_num = len(actual_table['cuid'].unique())
    
    return actual_table, actual_delivery_user_num

def _precision(actual_table):
    #return sum(actual_table['context_genre_num']) / sum(actual_table['context_num'])
    metrics_actual_table = actual_table
    # ユーザごとに算出して平均をとる
    user_actual_table = metrics_actual_table.groupby(['cuid'])['context_genre_num', 'context_num'].sum()
    user_actual_table['score'] = user_actual_table['context_genre_num'] / user_actual_table['context_num']
    user_actual_table = user_actual_table.reset_index()
    
    return sum(user_actual_table['score']) / len(user_actual_table['cuid'].unique())

def _recall(actual_table, test_table):
    #return sum(actual_table['context_genre_num']) / sum(test_table['context_genre_num'])
    metrics_actual_table = actual_table
    metrics_test_table = test_table
    # ユーザごとに算出して平均をとる
    user_recall_table = pd.DataFrame()
    user_recall_table['test_context_genre_num'] = metrics_test_table.groupby(['cuid'])['context_genre_num'].sum()
    user_recall_table['actual_context_genre_num'] = metrics_actual_table.groupby(['cuid'])['context_genre_num'].sum()
    
    # actual_tableに存在しないユーザのテスト期間データは無視する
    user_recall_table = user_recall_table.fillna(0)
    user_recall_table['score'] = user_recall_table['actual_context_genre_num'] / user_recall_table['test_context_genre_num']
    user_recall_table = user_recall_table.reset_index()

    return sum(user_recall_table['score']) / len(user_recall_table['cuid'])
    
# 実績をbooleanでカウントする場合
def _precision_boolean(actual_table):
    metrics_actual_table = actual_table
    # ユーザごとに算出して平均をとる
    # scoreが0以上であれば実績が1以上
    metrics_actual_table['score_boolean'] = metrics_actual_table['context_genre_num'].where(metrics_actual_table['context_genre_num']==0, 1)
    
    user_precision_table = pd.DataFrame()
    user_precision_table['context_genre_num'] = metrics_actual_table.groupby(['cuid'])['score_boolean'].sum()
    user_precision_table['context_num'] = metrics_actual_table.groupby(['cuid'])['score_boolean'].count()
    user_precision_table['score'] = user_precision_table['context_genre_num'] / user_precision_table['context_num']
    user_precision_table = user_precision_table.reset_index()

    return sum(user_precision_table['score']) / len(user_precision_table['cuid'].unique())

def _recall_boolean(actual_table, test_table):
    metrics_actual_table = actual_table
    metrics_test_table = test_table
    # ユーザごとに算出して平均をとる
    # scoreが0以上であれば実績が1以上
    metrics_test_table['context_genre_num_boolean'] = metrics_test_table['context_genre_num'].where(metrics_test_table['context_genre_num']==0, 1)
    metrics_actual_table['context_genre_num_boolean'] = metrics_actual_table['context_genre_num'].where(metrics_actual_table['context_genre_num']==0, 1)
    
    user_recall_table = pd.DataFrame()
    user_recall_table['test_context_genre_num'] = metrics_test_table.groupby(['cuid'])['context_genre_num_boolean'].sum()
    user_recall_table['actual_context_genre_num'] = metrics_actual_table.groupby(['cuid'])['context_genre_num_boolean'].sum()
    
    # actual_tableに存在しないユーザのテスト期間データは0とする
    user_recall_table = user_recall_table.fillna(0)
    user_recall_table['score'] = user_recall_table['actual_context_genre_num'] / user_recall_table['test_context_genre_num']
    user_recall_table = user_recall_table.reset_index()

    return sum(user_recall_table['score']) / len(user_recall_table['cuid'])

def _auc(recommend_table, actual_table):
    """
    基本的な考え方
    recommend_tableのある時間帯/状況のスコアが例えば20の時，
    その時間帯/状況の実績テーブルの，
    利用回数（context_genre_num）が2，
    状況発生回数（context_num）が5の時，
    true_list = [1,1,0,0,0]
    score_list = [20,20,20,20,20]
    を作成する
    """
    
    all_true_list = []
    all_scores_list = []
    
    try:
        auc_recommend_table = recommend_table.drop(['context_genre_num', 'context_num'], axis=1)
    except:
        auc_recommend_table = recommend_table
    auc_actual_table = actual_table.drop('score', axis=1)
    
    #column = ['cuid', 'time', 'poi_type', 'context']
    column = ['cuid', 'time', 'poi_type']
    auc_table = pd.merge(auc_recommend_table, auc_actual_table, on=column, how='inner')
    
    auc_table['context_genre_num'] = auc_table['context_genre_num'].astype(int)
    auc_table['context_num'] = auc_table['context_num'].astype(int)
    
    score_list = auc_table['score'].values.tolist()
    context_genre_num_list = auc_table['context_genre_num'].values.tolist()
    context_num_list = auc_table['context_num'].values.tolist()
    

    for s, g, c in zip(score_list, context_genre_num_list, context_num_list):
        true_list = [1 for t in range(g)]
        zero_num = c - g
        zero_list = [0 for z in range(zero_num)]
        true_list.extend(zero_list)

        scores_list = []
        scores_list.append(s)
        scores_list = scores_list * c

        all_true_list.extend(true_list)
        all_scores_list.extend(scores_list)
        
    fpr, tpr, thresholds = roc_curve(np.array(all_true_list), np.array(all_scores_list))
    auc_ = auc(fpr, tpr)
    
    precision_, recall_, pr_thresholds = precision_recall_curve(np.array(all_true_list), np.array(all_scores_list))
    ave_precision = average_precision_score(np.array(all_true_list), np.array(all_scores_list))
    return auc_, fpr, tpr, thresholds, all_true_list, all_scores_list, precision_, recall_, pr_thresholds, ave_precision

def _auc_boolean(recommend_table, actual_table):
    """
    基本的な考え方
    recommend_tableのある時間帯/状況のスコアが例えば20の時，
    その時間帯/状況の実績テーブルの，
    利用回数（context_genre_num）が2，
    状況発生回数（context_num）が5の時，
    true_list = [1,1,0,0,0]
    score_list = [20,20,20,20,20]
    を作成する
    """
    
    all_true_list = []
    all_scores_list = []
    
    auc_recommend_table = recommend_table.drop(['context_genre_num', 'context_num'], axis=1)
    auc_actual_table = actual_table.drop('score', axis=1)
    
    #column = ['cuid', 'time', 'poi_type', 'context']
    column = ['cuid', 'time', 'poi_type']
    actual_table = pd.merge(actual_table, context_count_table, on=column, how='inner')
    auc_table = pd.merge(auc_recommend_table, auc_actual_table, on=column, how='inner')
    
    auc_table['score_boolean'] = auc_table['context_genre_num'].where(auc_table['context_genre_num']==0, 1)
    auc_table['score_boolean'] = auc_table['score_boolean'].astype(int)
    
    all_scores_list = auc_table['score'].values.tolist()
    all_true_list = auc_table['score_boolean'].values.tolist()
    
    fpr, tpr, thresholds = roc_curve(np.array(all_true_list), np.array(all_scores_list))
    auc_ = auc(fpr, tpr)
    
    precision_, recall_, pr_thresholds = precision_recall_curve(np.array(all_true_list), np.array(all_scores_list))
    ave_precision = average_precision_score(np.array(all_true_list), np.array(all_scores_list))
    return auc_, fpr, tpr, thresholds, all_true_list, all_scores_list, precision_, recall_, pr_thresholds, ave_precision

def _delivery_user(recommend_table, score_threshold):
    return len(recommend_table[recommend_table['score'] >= score_threshold]['cuid'].unique())
    

def model_score(recommend_table, score_threshold, test_table, test_context_count_table, topk=None, is_boolean=False):
    
    # レコメンドテーブルの大きさが０など例外をキャッチする
    
    try:
        actual_table, actual_delivery_user_num = _get_actual_table(recommend_table, score_threshold, test_table, test_context_count_table, topk)
        
        if is_boolean:
            precision = _precision_boolean(actual_table)
            recall = _recall_boolean(actual_table, test_table)
            auc, fpr, tpr, thresholds, true_list, scores_list, precisions, recalls, pre_thresholds, ave_precision = _auc_boolean(recommend_table, actual_table)
        else:
            precision = _precision(actual_table)
            recall = _recall(actual_table, test_table)
            auc, fpr, tpr, thresholds, true_list, scores_list, precisions, recalls, pre_thresholds, ave_precision = _auc(recommend_table, actual_table)
            
        delivery_user_num = _delivery_user(recommend_table, score_threshold)
    except:
        raise Exception
    
    #return delivery_user, auc, precision, recall
    return delivery_user_num, actual_delivery_user_num, auc, fpr, tpr, thresholds, true_list, scores_list, precisions, recalls, pre_thresholds, ave_precision, precision, recall