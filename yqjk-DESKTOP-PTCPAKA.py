# -*- coding: UTF-8 -*-

import time
from flask import Flask, render_template, json, jsonify, request
import random
import datetime
from datetime import datetime

from pymysql import connect,cursors
 
class MysqlHelper:
    def __init__(self,
        host = "139.217.229.72",
        user = "root",
        password = "root",
        database = "bbs_talk",
        charset = 'utf8',
        port = 13306):

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self._conn = None
        self._cursor = None
 
    def _open(self):
        self._conn = connect(host=self.host,
               port=self.port,
               user=self.user,
               password=self.password,
               database=self.database,
               charset=self.charset)
        self._cursor = self._conn.cursor(cursors.DictCursor)
 
    def _close(self):
        self._cursor.close()
        self._conn.close()
 
    def one(self, sql, params=None):
        result: tuple = None
        try:
            self._open()
            self._cursor.execute(sql, params)
            result = self._cursor.fetchone()
        except Exception as e:
            print(e)
        finally:
            self._close()
        return result
 
    def all(self, sql, params=None):
        result: tuple = None
        try:
            self._open()
            self._cursor.execute(sql, params)
            result = self._cursor.fetchall()
        except Exception as e:
            print(e)
        finally:
            self._close()
        return result

 
async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# app.config['SQLALCHEMY_DATABASE_URI']='mysql+pymysql://root:root@mysql:3306/bbs_talk' 
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# 主页
@app.route('/yqjk')
def home_index():
    return render_template('rdgz.html') 
# 热点关注
@app.route('/yqjk/rdgz_index')
def rdgz_index():
    return render_template('rdgz.html')    
# 情感分析
@app.route('/yqjk/qgfx_index')
def qgfx_index():
    return render_template('qgfx.html')
# 主题检索
@app.route('/yqjk/ztjs_index')
def ztjs_index():
    return render_template('ztjs.html')
# 互动体验
@app.route('/yqjk/hdty_index')
def hdty_index():
    return render_template('hdty.html')

def hour_data(sql):
    hour_data = {}
    for idx in range(24):
        if idx < 10:
            hour_data["0"+str(idx)] = 0
        else:
            hour_data[str(idx)] = 0

    results = MysqlHelper().all(sql)

    for res in results:
        hour_data[res["hour"]] = res["cnt"]
        
    return list(hour_data.values())

@app.route('/emotion_dist', methods=['GET'])
def emotion_dist():
    sql_label_init = "select date_format(cm_launch_tme,'%H') as hour, count(*) as cnt from bbs_02_content_add_score where date(cm_launch_tme)='2019-08-01' and label={} and module='交通银行讨论区' group by hour order by hour"
    sql_label00 = sql_label_init.format(0.0)
    sql_label05 = sql_label_init.format(0.5)
    sql_label10 = sql_label_init.format(1.0)
    x_data = []
    for idx in range(24):
        if idx < 10:
            x_data.append("0"+str(idx))
        else:
            x_data.append(str(idx))

    data_00 = hour_data(sql_label00)
    data_05 = hour_data(sql_label05)
    data_10 = hour_data(sql_label10)

    return jsonify({
        'emotion_dist': {
            "x_data": x_data,
            "y_data": [data_00, data_05, data_10]
        }
    })


@app.route('/bbs_stat', methods=['GET'])
def bbs_stat():
    selected_date = request.args.get("selected_date")

    if selected_date == "":
        dt=datetime.now() #创建一个datetime类对象
        selected_date=dt.strftime('%Y-%m-%d')
    sql_data='''
        select date_format(cm_launch_tme,'%H') as hour,count(1) as cnt from bbs_02_content_week where date(cm_launch_tme)='{}' group by 1 order by 1
        '''
    
    results = MysqlHelper().all(sql_data.format(selected_date))
    
    mth_list=[]
    cnt_list=[]

    for res in results:
        mth_list.append(res["hour"])
        cnt_list.append(res["cnt"])
    ret=[mth_list,cnt_list]

    return jsonify(ret)

@app.route('/hot_list', methods=['GET'])
def hot_list():
    dt=datetime.now()
    etl_dte=dt.strftime('%Y-%m-%d')
    sql_hotlist = "select theme,sub_url,sum(case when substr(cm_launch_tme,1,10)='{}' then 1 else 0 end) as cnt1,count(*) as cnt,date_format(min(cm_launch_tme),'%Y-%m-%d %T') as min_cm_launch_tme, date_format(max(cm_launch_tme),'%Y-%m-%d %T') as max_cm_launch_tme from bbs_talk.bbs_02_content_week where module='交通银行讨论区' group by theme,sub_url order by cnt1 desc limit 4"
    results = MysqlHelper().all(sql_hotlist.format(etl_dte))

    ret = []
    for res in results:
        ret.append({
            'theme': res["theme"],
            'url': res["sub_url"],
            'count_today': str(res["cnt1"]),
            'cnt': res["cnt"],
            'min_cm_launch_tme': res["min_cm_launch_tme"],
            'max_cm_launch_tme': res["max_cm_launch_tme"]
        })
    return jsonify(ret)


@app.route('/hour_hot_list', methods=['GET'])
def hour_hot_list():
    selected_hour = request.args.get("selected_hour")
    selected_date = request.args.get("selected_date")
    if selected_date == "":
        dt=datetime.now() #创建一个datetime类对象
        selected_date=dt.strftime('%Y-%m-%d')

    sql_hotlist='''
                select theme,sub_url,date_format(cm_launch_tme,'%T') as cm_launch_tme,remark1 from bbs_02_content_week where date(cm_launch_tme)="{}" and hour(cm_launch_tme)="{}" and module="交通银行讨论区" order by cm_launch_tme desc
                '''  

    results = MysqlHelper().all(sql_hotlist.format(selected_date,selected_hour))

    ret = []
    for res in results:
        ret.append({
            'theme': res["theme"],
            'url': res["sub_url"],
            'cm_launch_tme': res["cm_launch_tme"],
            'remark': res["remark1"]
        })
    return jsonify(ret)


if __name__ == '__main__':
    app.run()