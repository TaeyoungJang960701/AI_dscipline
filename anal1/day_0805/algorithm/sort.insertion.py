# 병합 정렬은 분할 정복 (Devide and Conquer) 기법과 
# 재귀 알고리즘을 이용해서 정렬 알고리즘입니다. 
# 즉, 주어진 배열을 원소가 하나 밖에 남지 않을 때까지 
# 계속 둘로 쪼갠 후에 다시 크기 순으로 재배열 하면서 원래 크기의 배열로 합칩니다.

def merge_sort(a):
    n = len(a)
    if n <= 1:
        return a
    
    mid = n // 2
    g1 = merge_sort(a[:mid])
    # print(g1)
    g2 = merge_sort(a[mid:])
    # print(g2)

    # 두 그룹을 하나로 합침
    result = []                 # 합친 결과 최종 기억
    while g1 and g2 :           # 두 그룹의 요소값이 있는 동안 계속 반복
        if g1[0] < g2[0]:       # 두 그룹의 맨 앞 자료 비교
            result.append(g1.pop(0))
        else:
            result.append(g2.pop(0))

    while g1:
        result.append(g1.pop(0))
    while g2:
        result.append(g2.pop(0))
    return result
        # print(merge_sort(d))



d = [6, 8, 3, 1, 2, 4, 7, 5]
print(merge_sort(d))

print('-----------------')
def merge_sort2(a):
    n = len(a)
    if n <= 1:
        return 
    mid = n // 2
    g1 = a[:mid]
    g2 = a[mid:]
    merge_sort2(g1) # 계속 반으로 나누다가 길이가 1이 되면 쪼개기 멈춤
    merge_sort2(g2)
    # 두 그룹을 하나로 합치기
    i1 = 0
    i2 = 0
    ia = 0
    while i1 < len(g1) and i2 < len(g2):
        if g1[i1] < g2[i2]:  # 두 집합의 앞쪽 값들을 하나씩 비교해 
                             # 더 작은 a에 차례로 채우기
            a[ia] = g1[i1]
            i1 += 1
            ia += 1
        else:
            a[ia] = g2[i2]
            i2 += 1
            ia += 1

    # 아직 남아있는 자료들을 추가
    while i1 < len(g1):
        a[ia] = g1[i1]
        i1 += 1
        ia += 1
    while i2 < len(g2):
        a[ia] = g2[i2]
        i2 += 1
        ia += 1


d = [6, 8, 3, 1, 2, 4, 7, 5]
merge_sort2(d)
print('두 번째 방법\n',d,'\n-----------------')

print('세 번째 방법 : 두번째 방법을 값을 반환하는 방법으로 변환')
def merge_sort3(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort3(arr[:mid])
    right = merge_sort3(arr[mid:])
    result = []
    i = j = 0

    # 병합
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 남은 요소값 추가
    result += left[i:]
    result += right[j:]
    return result



d = [6, 8, 3, 1, 2, 4, 7, 5]
print(merge_sort3(d))