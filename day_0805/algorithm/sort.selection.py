# 선택 정렬(Selection Sort)은 주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로
# 알고리즘 과정: 
# 최소값 찾기: 정렬되지 않은 부분에서 가장 작은 값을 찾습니다.
# 교환: 찾은 최소값을 정렬되지 않은 부분의 맨 앞으로 이동시킵니다.
# 반복 : 정렬되지 않은 부분의 크기가 1이 될 때까지 위 과정을 반복합니다.

# 방법1 : 원리 이해를 우선
def find_minFunc(a):  # 진짜 함수로 정의해버려서 나열하는거야
    n = len(a)
    min_idx = 0
    for i in range(1, n):
        if a[i] < a[min_idx]:
            min_idx = i
    return min_idx


def sel_sort(a):
    result = []
    while a:
        min_idx = find_minFunc(a)
        value = a.pop(min_idx)
        result.append(value)
    return result


d = [2, 4, 5, 1, 3]
# print(find_minFunc(d))
print("선택 정렬 결과(sel_sort):", sel_sort(d.copy()))  # 🔧 d.copy() 사용으로 원본 유지


# 방법2 : 일반적 정렬 알고리즘을 구사 : result 안씀
# 각 반복마다 가장 작은 값을 해당 집합내의 맨 앞자리와 값을 바꿈
# selection sort는 사람이 보고 이해하긴 쉽지만 처리 속도가 느리다
def sel_sort2(a):
    n = len(a)
    for i in range(0, n - 1):   # 0부터 n-2까지 반복
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]


def find_insFunc(r, v):
    # 이미 정렬된 r의 자료를 앞에서부터 차례로 확인해가며 작업
    for i in range(0, len(r)):
        if v < r[i]:
            return i
    return len(r)  # v가 r의 모든 요소값보다 클 경우에는 맨 뒤에 삽입


def ins_sort(a):
    result = []
    while a:
        value = a.pop(0)
        ins_idx = find_insFunc(result, value)
        result.insert(ins_idx, value)   # 찾은 위치에 값을 삽입 또는 추가한다
        print(result)
    return result

d = [2, 4, 5, 1, 3]
print("삽입 정렬 중간 결과:")
ins_sort(d.copy())  # 🔧 d.copy() 사용해서 원본 d 유지
