# 삽입 정렬은 배열을 정렬된 부분과 정렬되지 않은 부분으로 나누어, 
# 정렬되지 않은 요소를 하나씩 정렬된 부분에 "삽입"하며 정렬을 완성하는 알고리즘입니다.
# 방법1 : 원리 이해를 우선
def find_insFunc(r,v):
    # 이미 정렬된 r의 자료를 앞에서부터 차례로 확인해가며 작업
    for i in range(0,len(r)):
        if v < r[i]:
            return i 
    return len(r)  # v가 r의 모든 요소값보다 클 경우에는 맨 뒤에 삽입

def ins_sort(a):
    result = []
    while a:
        value = a.pop(0)
        ins_idx = find_insFunc(result, value)
        result.insert(ins_idx,value)   # 찾은 위치에 값을 삽입 또는 추가한다
        print(result)
    return result

# 방법 2 : 일반적 정렬 알고리즘을 구사 : result x
def ins_sort2(a):
    n = len(a)
    for i in range(1,n):    # 두번째 값(인덱스1)부터 마지막까지 차례대로 '삽입할 대상' 선택
                            # 이 범위는 1부터 n-1까지임
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key: # key값보다 큰 값을 우측으로 밀기(참)
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
        print(a)


d = [2, 4, 5, 1, 3]
print(ins_sort2(d))