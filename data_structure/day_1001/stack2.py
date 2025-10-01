# 자료 구조중 Queue : FIFO 구조
from collections import deque

class Queue:
    def __init__(self, iterable = None):
        # queue 는 양쪽 끝에서 삽입 / 삭제가 O(1)으로 빠르게 처리
        self._data = deque()

        if iterable is not None:
            for x in iterable:
                self.enqueue(x)
            
    def enqueue(self, x):
        self._data.append(x)    # 뒤(rear)에 원소 추가
        return x
    
    def dequeue(self):          # 앞(front)에 원소 제거
        if not self._data:
            raise IndexError('dequeue from empty queue')
        return self._data.popleft()

    def front(self):    # 큐에서 맨 앞 원소 확인용 매소드
        if not self._data:
            raise IndexError('dequeue from empty queue')
        return self._data[0]
    
    def is_empty(self):
        return not self._data
    
    def size(self):
        return len(self._data)
    
    def clear(self):
        self._data.clear()

    def __repr__(self):     # 객체를 문자열로 표현할 때 사용
        return f'Queue(front -> back {list(self._data)})'
    
def demo_fifo():
    q = Queue()
    for item in ['a','b','c','d']:
        q.enqueue(item)
        print(f'enqueue {item} -> ', q)
    print('\nDequeue until empty (FIFO)')
    while not q.is_empty():
        print(f'dequeue -> ', q.dequeue(), ' | now :', q)

demo_fifo()
