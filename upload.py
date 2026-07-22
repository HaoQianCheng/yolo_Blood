import paramiko
import sys

src = '/mnt/g/WSL/model/best_rk3576.rknn'
dst = '/root/blood_label/best_n2_rk3576.rknn'
host = '192.168.135.137'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username='root', password='')

sftp = ssh.open_sftp()
sftp.put(src, dst)
sftp.close()
ssh.close()
print('上传完成')
