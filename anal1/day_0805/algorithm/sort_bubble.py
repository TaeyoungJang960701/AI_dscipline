# 버블 정렬은 인접한 두 개의 원소를 비교하여 자리를 교환하는 방식이다

def bubble_sort(a):
    while True:
        n = len(a)
        changed = False   # 자료를 바꿧는지 여부
        for i in range(0, n-1):
            if a[i] > a[i + 1]:   # 앞이 뒤보다 크면  바꿈
                print(a)
                a[i], a[i + 1] = a[i + 1], a[i]
                changed = True
        if changed == False:
            return
d = [2,4,5,1,3]
bubble_sort(d)
print(d)