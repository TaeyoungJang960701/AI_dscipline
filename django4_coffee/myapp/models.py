from django.db import models

# Create your models here.
# 이거 python .\manage.py inspectdb > aa.py 해서 만들어진 aa.py 파일에서 가져온거야
# 이게 잇어야 ORM을 쓸 수 잇대
class Survey(models.Model):
    rnum = models.AutoField(primary_key=True)
    gender = models.CharField(max_length=4, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    co_survey = models.CharField(max_length=10, blank=True, null=True)

    class Meta:
        managed = False         # 테이블 생성 안하겠다. 기존 테이블 쓸게
        db_table = 'survey'     # MariaDB의 테이블명

# ORM의 특징 -> 쟝고의 소스는 바꾸지 않겠다

# python manage.py makemigrations 이거는 모델스.py랑 연관이 잇대