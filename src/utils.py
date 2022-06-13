import os
import sys

class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

# NOTE: Write the lines in main.py file to log out
# log_out = args.log_dir + '/output.log'
# sys.stdout = Logger(log_out)

""" Heap queue algorithm (a.k.a. priority queue).

        Heaps are arrays for which a[k] <= a[2*k+1] and a[k] <= a[2*k+2] for all k, counting elements from 0.  
        For the sake of comparison, non-existing elements are considered to be infinite.  
        The interesting property of a heap is that a[0] is always its smallest element.
        Usage:
            heap = []            # creates an empty heap
            heappush(heap, item) # pushes a new item on the heap
            item = heappop(heap) # pops the smallest item from the heap
            item = heap[0]       # smallest item on the heap without popping it
            heapify(x)           # transforms list into a heap, in-place, in linear time
            item = heapreplace(heap, item) # pops and returns smallest item, and adds
                                        # new item; the heap size is unchanged

        This differs from textbook heap algorithms as follows:
        - We use 0-based indexing.  This makes the relationship between the
        index for a node and the indexes for its children slightly less
        obvious, but is more suitable since Python uses 0-based indexing.
        - Our heappop() method returns the smallest item, not the largest.
        These two make it possible to view the heap as a regular Python list
        without surprises: heap[0] is the smallest item, and heap.sort()
        maintains the heap invariant!
"""

__all__ = ['heappush', 'heappop', 'heapify', 'heapreplace', 'merge',
           'nlargest', 'nsmallest', 'heappushpop']

# from itertools import islice, repeat, count, imap, izip, tee
# from operator import itemgetter, neg
# import bisect

# def heappush(heap, item):

#     """Push item onto heap, maintaining the heap invariant."""
#     heap.append(item)
#     _siftdown(heap, 0, len(heap)-1)

# def heappop(heap):

#     """Pop the smallest item off the heap, maintaining the heap invariant."""
    
#     # Raises appropriate IndexError if heap is empty
#     lastelt = heap.pop()    
    
#     if heap:
#         returnitem = heap[0]
#         heap[0] = lastelt
#         _siftup(heap, 0)
    
#     else:
#         returnitem = lastelt
    
#     return returnitem

# def heapreplace(heap, item):
    
#     """Pop and return the current smallest value, and add the new item.
#     This is more efficient than heappop() followed by heappush(), and can be
#     more appropriate when using a fixed-size heap.  Note that the value
#     returned may be larger than item!  That constrains reasonable uses of
#     this routine unless written as part of a conditional replacement:
    
#         if item > heap[0]:
#             item = heapreplace(heap, item)
    
#     """
    
#     # Raises appropriate IndexError if heap is empty
#     returnitem = heap[0]    
#     heap[0] = item
#     _siftup(heap, 0)
    
#     return returnitem

# def heappushpop(heap, item):
    
#     """Fast version of a heappush followed by a heappop."""
    
#     if heap and heap[0] < item:
#         item, heap[0] = heap[0], item
#         _siftup(heap, 0)
    
#     return item

# def heapify(x):
    
#     """Transform list into a heap, in-place, in O(len(heap)) time."""
    
#     n = len(x)
    
#     # Transform bottom-up.  The largest index there's any point to looking at
#     # is the largest with a child index in-range, so must have 2*i + 1 < n,
#     # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
#     # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
#     # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    
#     for i in reversed(xrange(n // 2)):
#         _siftup(x, i)

# def nlargest(n, iterable):
    
#     """Find the n largest elements in a dataset.
#     Equivalent to:  sorted(iterable, reverse = True)[:n]
#     """
    
#     it = iter(iterable)
#     result = list(islice(it, n))
    
#     if not result:
#         return result
    
#     heapify(result)
#     _heappushpop = heappushpop
    
#     for elem in it:
#         _heappushpop(result, elem)
    
#     result.sort(reverse = True)
#     return result

# def nsmallest(n, iterable):
    
#     """Find the n smallest elements in a dataset.
#     Equivalent to:  sorted(iterable)[:n]
#     """
    
#     if hasattr(iterable, '__len__') and n * 10 <= len(iterable):
    
#         # For smaller values of n, the bisect method is faster than a minheap.
#         # It is also memory efficient, consuming only n elements of space.
#         it = iter(iterable)
#         result = sorted(islice(it, 0, n))
    
#         if not result:
#             return result
    
#         insort = bisect.insort
#         pop = result.pop
#         los = result[-1]    # los --> Largest of the nsmallest
    
#         for elem in it:
#             if los <= elem:
#                 continue
#             insort(result, elem)
#             pop()
#             los = result[-1]
    
#         return result
    
#     # An alternative approach manifests the whole iterable in memory but
#     # saves comparisons by heapifying all at once.  Also, saves time
#     # over bisect.insort() which has O(n) data movement time for every
#     # insertion.  Finding the n smallest of an m length iterable requires
#     #    O(m) + O(n log m) comparisons.
    
#     h = list(iterable)
#     heapify(h)
#     return map(heappop, repeat(h, min(n, len(h))))

# # 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# # is the index of a leaf with a possibly out-of-order value.  Restore the
# # heap invariant.

# def _siftdown(heap, startpos, pos):

#     newitem = heap[pos]

#     # Follow the path to the root, moving parents down until finding a place
#     # newitem fits.

#     while pos > startpos:

#         parentpos = (pos - 1) >> 1
#         parent = heap[parentpos]

#         if newitem < parent:
#             heap[pos] = parent
#             pos = parentpos
#             continue

#         break

#     heap[pos] = newitem

# def _siftup(heap, pos):
    
#     endpos = len(heap)
#     startpos = pos
#     newitem = heap[pos]
    
#     # Bubble up the smaller child until hitting a leaf.
    
#     childpos = 2*pos + 1    # leftmost child position
#     while childpos < endpos:
    
#         # Set childpos to index of smaller child.
#         rightpos = childpos + 1
    
#         if rightpos < endpos and not heap[childpos] < heap[rightpos]:
#             childpos = rightpos
    
#         # Move the smaller child up.
#         heap[pos] = heap[childpos]
#         pos = childpos
#         childpos = 2*pos + 1
    
#     # The leaf at pos is empty now.  Put newitem there, and bubble it up
#     # to its final resting place (by sifting its parents down).
#     heap[pos] = newitem
#     _siftdown(heap, startpos, pos)

# # If available, use C implementation
# try:
#     from _heapq import heappush, heappop, heapify, heapreplace, nlargest, nsmallest, heappushpop
# except ImportError:
#     pass

# def merge(*iterables):

#     '''Merge multiple sorted inputs into a single sorted output.
#     Similar to sorted(itertools.chain(*iterables)) but returns a generator,
#     does not pull the data into memory all at once, and assumes that each of
#     the input streams is already sorted (smallest to largest).
#     >>> list(merge([1,3,5,7], [0,2,4,8], [5,10,15,20], [], [25]))
#     [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]
#     '''
    
#     _heappop, _heapreplace, _StopIteration = heappop, heapreplace, StopIteration

#     h = []
#     h_append = h.append
    
#     for itnum, it in enumerate(map(iter, iterables)):
#         try:
#             next = it.next
#             h_append([next(), itnum, next])
#         except _StopIteration:
#             pass
    
#     heapify(h)

#     while 1:
#         try:
#             while 1:
#                 v, itnum, next = s = h[0]   # raises IndexError when h is empty
#                 yield v
#                 s[0] = next()               # raises StopIteration when exhausted
#                 _heapreplace(h, s)          # restore heap condition
#         except _StopIteration:
#             _heappop(h)                     # remove empty iterator
#         except IndexError:
#             return

# # Extend the implementations of nsmallest and nlargest to use a key= argument

# _nsmallest = nsmallest
# def nsmallest(n, iterable, key=None):
    
#     """Find the n smallest elements in a dataset.
#     Equivalent to:  sorted(iterable, key=key)[:n]
#     """
    
#     if key is None:
    
#         it = izip(iterable, count())                        # decorate
#         result = _nsmallest(n, it)
#         return map(itemgetter(0), result)                   # undecorate
    
#     in1, in2 = tee(iterable)
#     it = izip(imap(key, in1), count(), in2)                 # decorate
#     result = _nsmallest(n, it)
    
#     return map(itemgetter(2), result)                       # undecorate

# _nlargest = nlargest
# def nlargest(n, iterable, key=None):
    
#     """Find the n largest elements in a dataset.
#     Equivalent to:  sorted(iterable, key=key, reverse=True)[:n]
#     """
    
#     if key is None:
    
#         it = izip(iterable, imap(neg, count()))             # decorate
#         result = _nlargest(n, it)
#         return map(itemgetter(0), result)                   # undecorate
    
#     in1, in2 = tee(iterable)
#     it = izip(imap(key, in1), imap(neg, count()), in2)      # decorate
#     result = _nlargest(n, it)
    
#     return map(itemgetter(2), result)                       # undecorate