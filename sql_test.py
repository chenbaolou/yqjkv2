# -*- coding: UTF-8 -*-

import time
from flask import Flask, render_template, json, jsonify, request
import random
import datetime
from datetime import datetime
import requests
import json

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

 
def hot_list():
    dt=datetime.now()
    etl_dte=dt.strftime('%Y-%m-%d')
    etl_dte = "2019-10-20"
    sql_hotlist = "select theme,sub_url,sum(case when substr(cm_launch_tme,1,10)='{}' then 1 else 0 end) as cnt1,count(*) as cnt,date_format(min(cm_launch_tme),'%Y-%m-%d %T') as min_cm_launch_tme, date_format(max(cm_launch_tme),'%Y-%m-%d %T') as max_cm_launch_tme from bbs_talk.bbs_02_content_week where module='交通银行讨论区' group by theme,sub_url order by cnt1 desc limit 10"
    print(sql_hotlist.format(etl_dte))
    results = MysqlHelper().all(sql_hotlist.format(etl_dte))

    ret = []
    for res in results:
        if int(res["cnt1"]) > 0:
            ret.append({
                'theme': res["theme"],
                'url': res["sub_url"],
                'count_today': int(res["cnt1"]),
                'cnt': res["cnt"],
                'min_cm_launch_tme': res["min_cm_launch_tme"],
                'max_cm_launch_tme': res["max_cm_launch_tme"]
            })

    print(ret)

if __name__ == '__main__':
    hot_list()