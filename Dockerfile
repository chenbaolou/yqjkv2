FROM python:3.6.7

RUN mkdir -p /home/project/yqjk
WORKDIR /home/project/yqjk
COPY requirements.txt in /home/project/yqjk
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
COPY . /home/project/yqjk