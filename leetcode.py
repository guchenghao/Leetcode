#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\leetcode.py
# Project: d:\CodeWareHouse
# Created Date: Wednesday, February 7th 2018, 4:01:41 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Saturday, 10th March 2018 12:09:38 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###


# %%
# * two sum
# ! 如果使用冒泡排序的方法，会超时
import numpy as np


def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    for i in range(len(nums)):
        number = nums[i]
        nums_1 = nums[i + 1:]
        nums_1 = np.array(nums_1)
        sum_ = nums_1 + number
        p = (sum_ == target)
        if 1 in p:
            p = list(p)
            idx = p.index(True)
            return [i, i + idx + 1]
        else:
            continue


# %%
# * Remove Element
def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    j = 0
    for i in range(len(nums)):
        if nums[i] == val:
            continue
        else:
            nums[j] = nums[i]
            j = j + 1

    return j


# %%
# * Search Insert Position
def searchInsert(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if target in nums:
        return nums.index(target)
    else:
        for i in range(len(nums)):
            if target < nums[i]:
                return i
            if target > nums[i]:
                if i == len(nums) - 1:
                    return i + 1
                continue


# %%
# * Missing Number
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()

    for i in range(len(nums)):
        if nums[0] != 0:
            return 0
        if nums[-1] != len(nums):
            return len(nums)
        if nums[i+1] - nums[i] == 1:
            continue
        else:
            return nums[i] + 1


# %%
# * Remove Duplicates from Sorted List
def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if head is None or head.next is None:
        return head
    seen = [head.val]
    prev_node = dum = head
    node = head.next
    while node is not None:
        if node.val in seen:
                # delete it
            prev_node.next = node.next
            node = node.next
        else:
            seen.append(node.val)
            prev_node = node
            node = node.next

    return dum


# %%
# * Length of Last Word
def lengthOfLastWord(self, s):
    """
    :type s: str
    :rtype: int
    """
    strArr = s.strip(' ').split(' ')
    return len(strArr[len(strArr) - 1])


# %%
# * Path Sum
# * 使用了递归的方式,遍历树
def hasPathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if root is None:
        return False

    if sum - root.val == 0 and root.left is None and root.right is None:
        return True
    else:
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)


# %%
# * Contains Duplicate
def containsDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    nums.sort()
    if len(nums) == 1:
        return False
    for i in range(len(nums)):
        if i == len(nums) - 1:
            break
        if nums[i] == nums[i + 1]:
            return True
        else:
            continue
    return False


# %%
# * Contains Duplicate II
def containsNearbyDuplicate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    dic = {}
    for i, v in enumerate(nums):
        if v in dic and i - dic[v] <= k:
            return True
        dic[v] = i
    return False


# %%
# * Best Time to Buy and Sell Stock
# def maxProfit(self, prices):  # ! 超时
#     """
#     :type prices: List[int]
#     :rtype: int
#     """
#     result = []
#     for i in range(len(prices)):
#         if i == len(prices) - 1:
#             break
#         num_1 = prices[i + 1:]

#         if max(num_1) - prices[i] <= 0:
#             continue
#         else:
#             result.append(max(num_1) - prices[i])

#     if result == []:
#         return 0
#     else:
#         return max(result)
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    n = len(prices)
    if n <= 1:
        return 0
    max_profit = 0
    low_price = prices[0]
    for i in range(1, n):
        low_price = min(low_price, prices[i])
        max_profit = max(max_profit, prices[i]-low_price)
    return max_profit


# %%
# * Majority Element
import math


def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    num_dict = {}
    n = len(nums)
    if n == 1:
        return nums[0]
    for idn, v in enumerate(nums):
        if v in num_dict:
            num_dict[v] = num_dict[v] + 1
        else:
            num_dict[v] = 1
    result = max(num_dict.values())
    if result >= math.floor(n / 2):
        return max(num_dict, key=num_dict.get)


# %%
# * Find All Numbers Disappeared in an Array
def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return list(set(range(1, len(nums)+1)) - set(nums))


# %%
# * Reshape the Matrix
import numpy as np


def matrixReshape(self, nums, r, c):
    """
    :type nums: List[List[int]]
    :type r: int
    :type c: int
    :rtype: List[List[int]]
    """
    try:
        return np.reshape(nums, (r, c)).tolist()
    except:
        return nums


# %%
# * Largest Number At Least Twice of Others
import copy


def dominantIndex(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    copy_nums = copy.deepcopy(nums)
    nums.sort()
    maximum = nums[-1]
    for i in range(len(nums) - 1):
        if nums[i] == maximum:
            return -1
        else:
            if maximum >= nums[i] * 2:
                continue
            else:
                return -1
    return copy_nums.index(maximum)


# %%
# * Move Zeroes
def moveZeroes(self, nums):  # ! 这个方法慢
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    zero_num = nums.count(0)
    for i in range(zero_num):
        nums.remove(0)
        nums.append(0)


def moveZeroes(self, nums):  # ! 这个方法快100ms
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    length = len(nums)
    if length <= 1:
        return
    zero_count = 0
    i = 0
    while i < length:
        if nums[i] == 0:
            zero_count += 1
        else:
            nums[i - zero_count] = nums[i]
        i += 1
    j = 1
    while j <= zero_count:
        nums[length - j] = 0
        j += 1


# %%
# * Two Sum II - Input array is sorted
import numpy as np


def twoSum(self, numbers, target):  # ! 方法可行，但超时了
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    for idn, v in enumerate(numbers):
        number_1 = np.array(numbers[idn + 1:])
        result = number_1 + v
        p = (result == target)

        if 1 in p:
            p = list(p)
            idn_2 = p.index(True)
            return [idn + 1, idn_2 + idn + 2]


def twoSum(self, numbers, target):  # ! 这个方法可行
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    numbers_dict = {}
    for idn, v in enumerate(numbers):
        if target - v in numbers_dict:
            return [numbers_dict[target - v] + 1, idn + 1]
        numbers_dict[v] = idn


# %%
# * Max Consecutive Ones
from collections import defaultdict


def findMaxConsecutiveOnes(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums_dict = defaultdict(int)
    i = 0
    for num in nums:
        if num == 1:
            nums_dict[i] = nums_dict[i] + 1
        else:
            i = i + 1

    if nums_dict.values():
        return max(nums_dict.values())
    else:
        return 0


# %%
# * Maximum Product of Three Numbers
from functools import reduce


def maximumProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()

    return max(nums[1] * nums[0] * nums[-1], reduce(lambda x, y: x * y, nums[-3:]))


# %%
# * Longest Continuous Increasing Subsequence
from collections import defaultdict


def findLengthOfLCIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums_dict = defaultdict(int)
    count = 1
    i = 0
    if nums == []:
        return 0
    for idn in range(len(nums) - 1):
        if nums[idn + 1] > nums[idn]:
            count = count + 1
            nums_dict[i] = count
        else:
            i = i + 1
            count = 1

    if nums_dict.values():
        return max(nums_dict.values())
    else:
        return 1


# %%
# * Find Pivot Index
from functools import reduce


def pivotIndex(self, nums):  # ! 超时了
    """
    :type nums: List[int]
    :rtype: int
    """
    for idn, num in enumerate(nums):
        if idn == 0:
            if reduce(lambda x, y: x + y, nums[idn + 1:]) == 0:
                return idn
            else:
                continue
        if idn == len(nums) - 1:
            if reduce(lambda x, y: x + y, nums[:idn]) == 0:
                return idn
            else:
                break
        if reduce(lambda x, y: x + y, nums[idn + 1:]) == reduce(lambda x, y: x + y, nums[:idn]):
            return idn

    return -1


def pivotIndex(self, nums):  # ! 143ms
    """
    :type nums: List[int]
    :rtype: int
    """
    left = 0
    right = sum(nums)
    for idn, num in enumerate(nums):
        right -= num
        if right == left:
            return idn
        left += num
    return -1


# %%
# * Plus One
def plusOne(self, digits):  # ! 68ms
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    num = 0
    for item in digits:
        num = num * 10 + item

    return [int(i) for i in str(num+1)]


# %%
# * Pascal's Triangle
def generate(self, numRows):  # ! 最直接的方式，双重循环, O(n^2)
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    if numRows == 0:
        return []
    result = []
    temp = [0, 1]

    for i in range(numRows):
        row = []
        for j in range(len(temp) - 1):
            row.append(temp[j] + temp[j+1])
        result.append(row)
        temp = row[:]
        temp.insert(0, 0)
        temp.append(0)

    return result


def generate(self, numRows):
    """
    :type numRows: int
    :rtype: List[List[int]]
    :example:
              1 3 3 1 0
           +  0 1 3 3 1
           =  1 4 6 4 1
    """
    if numRows == 0:
        return []

    res = [[1]]
    # ! "+" 在这里表示数组的拼接
    for _ in range(1, numRows):
        res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
    return res


# %%
# * Pascal's Triangle II
def getRow(self, rowIndex):
    """
    :type rowIndex: int
    :rtype: List[int]
    """

    res = [[1]]

    for _ in range(1, rowIndex + 1):
        res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]

    return res[rowIndex]


def getRow(self, rowIndex):  # ! 改良版,更快, 63ms
    """
    :type rowIndex: int
    :rtype: List[int]
    """

    res = [1]

    for _ in range(1, rowIndex + 1):
        res = list(map(lambda x, y: x+y, res + [0], [0] + res))

    return res


def getRow(self, rowIndex):  # ! 更快 55ms
    """
    :type rowIndex: int
    :rtype: List[int]
    """

    res = [1]
    for _ in range(1, rowIndex + 1):
        res = [x + y for x, y in zip([0]+res, res+[0])]

    return res


# %%
# * Maximum Average Subarray I
def findMaxAverage(self, nums, k):  # ! 超时
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    maxSum = -10001
    curSum = 0
    for i in range(len(nums) - k + 1):
        curSum = sum(nums[i:i + k])
        maxSum = max(maxSum, curSum)
    return float(maxSum / k)


def findMaxAverage(self, nums, k):  # ! 212ms
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    sums = [0] + list(itertools.accumulate(nums))
    # ! sub: sums[k:] 和 sums对应位置相减，不对应的位置舍弃
    return max(map(operator.sub, sums[k:], sums)) / k


def findMaxAverage(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    begin = 0
    end = 0
    n = len(nums)
    kavg = 0
    if k < len(nums):
        ksum = sum(nums[0:k])  # sum of first k items
        kavg = ksum / float(k)  # avg of first k items
        beg = 0  # beginning of avg window
        x = k  # end of avg window

        while x < n:  # when end falls beyond n
            ksum = ksum - nums[beg] + nums[x]      # get ksum
            # set kavg to maximum of kavg and new kavg
            kavg = max(kavg, (ksum / float(k)))
            beg += 1                               # begining of avg window increases
            x += 1                               # end of avg window increases
    else:
        # kavg if items are lesser than k
        kavg = sum(nums) / float(len(nums))
    return kavg


# %%
# * Can Place Flowers
def canPlaceFlowers(self, flowerbed, n):  # ! 118ms
    """
    :type flowerbed: List[int]
    :type n: int
    :rtype: bool
    """
    count = 0
    w = len(flowerbed) - 1
    for idn, v in enumerate(flowerbed):
        if count >= n:
            return True
        if v == 0 and idn == 0:
            if idn == w:
                count += 1
                break
            if flowerbed[idn + 1] != 1:
                count += 1
                flowerbed[idn] = 1
                continue
            continue
        if v == 0 and idn == w:
            if flowerbed[idn - 1] != 1:
                count += 1
                break
            break

        if v == 0 and flowerbed[idn + 1] != 1 and flowerbed[idn - 1] != 1:
            flowerbed[idn] = 1
            count += 1

    return count >= n


# %%
# * Third Maximum Number
def thirdMax(self, nums):  # ! 64ms
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    num_set = list(set(nums))
    num_set.sort()
    if len(num_set) < 3:
        return num_set[-1]

    return num_set[-3]


# %%
# * Rotate Array
def rotate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    n = len(nums)
    k = k % n
    nums[:] = nums[n-k:] + nums[:n-k]


# %%
# * Shortest Unsorted Continuous Subarray
def findUnsortedSubarray(self, nums):  # ! 104ms
    """
    :type nums: List[int]
    :rtype: int
    """
    # ! 列表推导式
    res = [i for (i, (a, b)) in enumerate(
        zip(nums, sorted(nums))) if a != b]
    return 0 if not res else res[-1]-res[0]+1


# %%
# * Find All Duplicates in an Array
def findDuplicates(self, nums):  # ! 这种方式运行时间 322ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    nums.sort()
    result = []
    for i in range(len(nums) - 1):
        if nums[i] == nums[i + 1]:
            result.append(nums[i])

    return result


def findDuplicates(self, nums):  # ! 这种方式运行时间长 359ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    nums.sort()
    result = []
    i = 0
    while i < len(nums) - 1:
        if nums[i] == nums[i + 1]:
            result.append(nums[i])
            i = i + 2
        else:
            i = i + 1

    return result


# %%
# * Teemo Attacking
def findPoisonedDuration(self, timeSeries, duration):  # ! 138ms
    """
    :type timeSeries: List[int]
    :type duration: int
    :rtype: int
    """
    result = duration
    if timeSeries == []:
        return 0
    for i in range(len(timeSeries) - 1):
        if timeSeries[i + 1] > (timeSeries[i] + duration):
            result += duration
            continue
        else:
            result += (timeSeries[i + 1] - timeSeries[i])

    return result


# %%
# * Product of Array Except Self
def productExceptSelf(self, nums):  # ! 167ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    p = 1
    n = len(nums)
    output = []
    # ! [1,1,2,6]
    for i in range(0, n):
        output.append(p)
        p = p * nums[i]
    p = 1
    # ! [24,12,4,1]
    for i in range(n-1, -1, -1):
        output[i] = output[i] * p
        p = p * nums[i]
    return output  # ! [24,12,8,6]


# %%
# * Subsets
def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    res = [[]]
    for num in nums:
        res += [item+[num] for item in res]
    return res


# %%
def numJewelsInStones(self, J, S):
    """
    :type J: str
    :type S: str
    :rtype: int
    """
    return sum(map(J.count, S))  # ! map函数的用法
    # return sum(s in J for s in S)


# %%
# * Keyboard Row
import re


def findWords(self, words):
    """
    :type words: List[str]
    :rtype: List[str]
    """
    return list(filter(re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match, words))


# %%
# * Distribute Candies
def distributeCandies(self, candies):
    """
    :type candies: List[int]
    :rtype: int
    """
    number_sister = len(candies) / 2

    candies_set = set(candies)

    if len(candies_set) <= number_sister:
        return len(candies_set)
    else:
        return int(number_sister)


# %%
# * Island Perimeter
def islandPerimeter(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    result = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] is 1:
                result += 4
                if j > 0 and grid[i][j-1] is 1:
                    result -= 2
                if i > 0 and grid[i-1][j] is 1:
                    result -= 2
    return result


# %%
# * Single Number
def singleNumber(self, nums):  # ! 93ms
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    i = 0
    if len(nums) == 1:
        return nums[0]
    while i <= len(nums) - 1:
        if i == len(nums) - 1:
            return nums[i]
        if nums[i] == nums[i + 1]:
            i = i + 2
            continue
        else:
            return nums[i]


import operator


def singleNumber(self, nums):  # ! 68ms
    """
    :type nums: List[int]
    :rtype: int
    """
    # ! 使用异或运算，比较巧妙
    res = 0
    for num in nums:
        res ^= num
    return res


# %%
# * Find the Difference
def findTheDifference(self, s, t):  # ! 118ms 遍历字符串
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    return [item for item in t if item not in s or s.count(item) != t.count(item)][0]


import operator
from functools import reduce


def findTheDifference(self, s, t):  # ! 68ms 采用异或
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    return chr(reduce(operator.xor, map(ord, s + t)))


def findTheDifference(self, s, t):
    # ! 分别两个字符串数组的所有字符的AscII码值总和，再相减
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    sum1 = sum(map(ord, [c for c in s]))
    sum2 = sum(map(ord, [c for c in t]))
    return chr(sum2-sum1)


# %%
# * First Unique Character in a String
# ! 如果使用遍历字符串的方式，会超时
def firstUniqChar(self, s):  # ! 90ms
    """
    :type s: str
    :rtype: int
    """
    # ! 只需要遍历26次
    letters = 'abcdefghijklmnopqrstuvwxyz'
    index = [s.index(l) for l in letters if s.count(l) == 1]
    return min(index) if len(index) > 0 else -1


# %%
# * Minimum Index Sum of Two Lists
def findRestaurant(self, list1, list2):
    """
    :type list1: List[str]
    :type list2: List[str]
    :rtype: List[str]
    """
    # ! 将 list转化为字典
    H = {string: idx for (idx, string) in enumerate(list1)}
    m, items = len(list2) + len(list1), []
    for i, string in enumerate(list2):
        if string in H:
            if i + H[string] < m:
                m = i + H[string]
                items = [string]  # ! 替换原本的数组
            elif i + H[string] == m:  # ! 判断 Sum of indexs 相等的情况
                items.append(string)  # ! 把元素添加进数组
    return items


# %%
# * Reverse String
def reverseString(self, s):
    """
    :type s: str
    :rtype: str
    """
    s_Arr = [item for item in s]
    s_Arr.reverse()
    return "".join(s_Arr)


# %%
# * Valid Anagram
from collections import Counter


def isAnagram(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    s_dict = Counter(s)
    t_dict = Counter(t)

    if s_dict - t_dict == {} and t_dict - s_dict == {}:
        return True
    else:
        return False


# %%
# * Intersection of Two Arrays
def intersection(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    nums1_set = set(nums1)
    nums2_set = set(nums2)

    return list(nums1_set & nums2_set)


# %%
# * Intersection of Two Arrays II
from collections import Counter


def intersect(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    c1 = Counter(nums1)
    c2 = Counter(nums2)
    return list((c1 & c2).elements())


# %%
# * Array Partition I
def arrayPairSum(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    return sum(nums[::2])


# %%
# * Set Mismatch
def findErrorNums(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]


# %%
# * Happy Number
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    mem = set()
    while n != 1:
        n = sum([int(i) ** 2 for i in str(n)])
        if n in mem:
            return False
        else:
            mem.add(n)
    else:
        return True


# %%
# * Find All Anagrams in a String
from collections import Counter


def findAnagrams(self, s, p):  # ! 超时
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    p_dict = Counter(p)
    result = []
    # ! 因为每次循环都要调用Counter函数，所以总体时间会超时
    for i in range(len(s) - len(p) + 1):
        s_dict = Counter(s[i:i + len(p)])
        if s_dict - p_dict == {}:
            result.append(i)

    return result


def findAnagrams(self, s, p):  # ! 改良版 377ms
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    res = []
    pCounter = Counter(p)
    sCounter = Counter(s[:len(p)-1])
    for i in range(len(p)-1, len(s)):
        sCounter[s[i]] += 1   # include a new char in the window
        # This step is O(1), since there are at most 26 English letters
        if sCounter == pCounter:
            res.append(i-len(p)+1)   # append the starting index
        # decrease the count of oldest char in the window
        sCounter[s[i-len(p)+1]] -= 1
        if sCounter[s[i-len(p)+1]] == 0:
            del sCounter[s[i-len(p)+1]]   # remove the count if it is 0
    return res


# %%
# * Word Pattern
# ! 这道题目和Isomorphic Strings属于同类型题目
def wordPattern(self, pattern, str):
    """
    :type pattern: str
    :type str: str
    :rtype: bool
    """
    s = pattern
    t = str.split()
    return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)


# %%
# * Isomorphic Strings
def isIsomorphic(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))


# %%
# * Count Primes
def countPrimes(self, n):  # ! 厄拉多筛选法的变种
    """
    :type n: int
    :rtype: int
    """
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return sum(primes)


def countPrimes(self, n):  # ! 厄拉多筛选法 超时
    """
    :type n: int
    :rtype: int
    """
    if n < 3:
        return 0
    nums_list = list(range(1, n))
    nums_list[0] = 0
    for i in range(2, n):
        if nums_list[i-1] != 0:
            for j in range(i*2, n, i):
                nums_list[j-1] = 0
    result = [x for x in nums_list if x != 0]
    return len(result)


# %%
# * Delete Node in a Linked List
def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    # ! 从链表删除值为val的结点
    node.val = node.next.val
    node.next = node.next.next


# %%
# * Reverse Linked List
def reverseList(self, head):
    # ! 在遍历链表的时候，修改指针所指向的方向
    prev = None
    while head:
        curr = head
        curr.next = prev  # ! 调转指针方向
        prev = curr
        head = head.next
    return prev


# %%
# * Merge Two Sorted Lists
# ! 这种方法使用额外的空间，因为该程序创建了一个新的链表
def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    dummy = cur = ListNode(0)
    # ! 如果两个链表中有一个链表被遍历完了，就退出循环，剩下的节点直接接入cur链表的表尾
    while l1 and l2:
        # ! l1和l2两个链表中，值小的那个接到cur链表尾部
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


# %%
# * Linked List Cycle
def hasCycle(self, head):  # ! Floyd判圈法(龟兔赛跑算法)
    """
    :type head: ListNode
    :rtype: bool
    """
    try:
        slow = head  # ! 龟
        fast = head.next  # ! 兔
        while slow is not fast:
            slow = slow.next
            fast = fast.next.next
        return True

    except AttributeError as e:
        return False


def hasCycle(self, head):  # ! 这个更快些
    """
    :type head: ListNode
    :rtype: bool
    """
    cur = head
    while cur:
        if cur.next == head:
            return True
        n = cur.next
        cur.next = head
        cur = n
    return False


# %%
# * Remove Linked List Elements
def removeElements(self, head, val):
    """
    :type head: ListNode
    :type val: int
    :rtype: ListNode
    """
    # ! 首先加个链表头dummy， 方便遍历第一个结点
    dummy = ListNode(-1)
    dummy.next = head
    cur = dummy

    while cur is not None and cur.next is not None:
        if cur.next.val == val:
            cur.next = cur.next.next  # ! 删除操作
        else:
            cur = cur.next  # ! 遍历下一个结点

    return dummy.next  # ! 返回原链表的首结点


# %%
# * Intersection of Two Linked Lists
def getIntersectionNode(self, headA, headB):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    curA, curB = headA, headB
    lenA, lenB = 0, 0
    # ! 获取两个链表的长度
    while curA is not None:
        lenA += 1
        curA = curA.next
    while curB is not None:
        lenB += 1
        curB = curB.next
    # ! 让curA和curB指针复位
    curA, curB = headA, headB
    # ! 让指针在较长的链表中移两者的长度差
    if lenA > lenB:
        for i in range(lenA-lenB):
            curA = curA.next
    elif lenB > lenA:
        for i in range(lenB-lenA):
            curB = curB.next
    # !寻找两个链表中相同的结点
    while curB != curA:
        curB = curB.next
        curA = curA.next
    return curA


# %%
# * Self Dividing Numbers
def selfDividingNumbers(self, left, right):
    """
    :type left: int
    :type right: int
    :rtype: List[int]
    """
    def is_self_dividing(num): return '0' not in str(num) and all(
        [num % int(digit) == 0 for digit in str(num)])
    # ! 非常巧妙地利用了filter函数
    return list(filter(is_self_dividing, range(left, right + 1)))


# %%
# * Add Digits
def addDigits(self, num):
    """
    :type num: int
    :rtype: int
    """
    while len(str(num)) != 1:
        num = sum([int(item) for item in str(num)])

    else:
        return num


# %%
# * Excel Sheet Column Number
# ? A -> 1, B ->2
def titleToNumber(self, s):  # ! 这道题本质上是26进制转10进制
    """
    :type s: str
    :rtype: int
    """
    s = s[::-1]  # !将数组翻转，即reverse
    sum = 0
    for idn, char in enumerate(s):
        # ! ord函数用于计算字符的ASCII码值
        sum += (ord(char) - 65 + 1) * (26 ** idn)
    return sum


# %%
# * Power of Two
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """

    while n % 2 == 0 and n is not 0:
        n = n / 2

    else:
        if n == 1:
            return True

    return False


# %%
# * Power of Three
def isPowerOfThree(self, n):  # ! 这道题和上一题一模一样，没什么意义
    """
    :type n: int
    :rtype: bool
    """
    while n % 3 == 0 and n is not 0:
        n = n / 3

    else:
        if n == 1:
            return True

    return False


# %%
# * Ugly Number
def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return False

    for item in (2, 3, 5):
        while num % item == 0:
            num = num / item

    return num == 1


# %%
# * Valid Perfect Square
def isPerfectSquare(self, num):  # ! 这个方法其实作弊了，这个方法和使用sqrt函数没什么区别
    """
    :type num: int
    :rtype: bool
    """
    if num == 0:
        return False
    return True if int(num**0.5)*int(num**0.5) == num else False


def isPerfectSquare(self, num):  # !这个代码有问题，一直通过不了
    r = num
    while r*r > num:
        r = (r + num/r) / 2
    return r*r == num


def isPerfectSquare(self, num):  # !这个方法是可行的，使用了牛顿法
    """
    :type num: int
    :rtype: bool
    """
    r = 1.0
    while abs(r*r - num) > 0.0001:
        r = (r + num/r) / 2
    return int(r) * int(r) == num


# %%
# * Sqrt(x)
def mySqrt(self, x):  # !只保留整数部分
    """
    :type x: int
    :rtype: int
    """
    r = 1.0
    # ! 利用牛顿下山法开根号
    while abs(r*r - x) > 0.0001:
        r = (r + x/r) / 2
    return int(r)


# %%
# * Excel Sheet Column Title
def convertToTitle(self, n):
    """
    :type n: int
    :rtype: str
    """
    # ! 这个字符数组的索引从0开始
    capitals = [chr(x) for x in range(ord('A'), ord('Z')+1)]
    result = []
    while n > 0:
        result.append(capitals[(n-1) % 26])
        n = (n-1) // 26  # ! // 取整除 - 返回商的整数部分
    result.reverse()
    return ''.join(result)


def convertToTitle(self, n):  # ! 改进版，减少了存储空间和运行时间
    """
    :type n: int
    :rtype: str
    """
    result = []
    while n > 0:
        result.append(chr((n-1) % 26 + 65))
        n = (n-1) // 26
    result.reverse()
    return ''.join(result)


# %%
# * Reverse Integer
def reverse(self, x):
    """
    :type x: int
    :rtype: int
    """
    result = []
    for item in str(x):
        result.insert(0, item)

    if str(x).find('-') > -1:
        result.insert(0, result.pop())

    n = int("".join(result))

    return n if n.bit_length() < 32 else 0


# %%
# * Add Binary
def addBinary(self, a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    # ! bin函数是返回一个二进制字符串
    return bin(int(a, 2)+int(b, 2))[2:]


# %%
# * Nth Digit
def findNthDigit(self, n):
    """
    :type n: int
    :rtype: int
    """
    # ! 这道题的核心思想是通过划分整数区间，在区间内找到相应数字
    if n < 0:
        return 0
    count = 9
    start = 1
    length = 1
    while n > (length * count):
        n -= length * count  # ! 确定n所在的区间,例如：n=150就在(10, 99)这个区间中
        length += 1
        start *= 10
        count *= 10
    start += (n - 1)/length  # ! 确定n落在该区间那个数字上
    return int((str(start)[(n - 1) % length]))


# %%
# * Sum of Square Numbers
import math


def judgeSquareSum(self, c):
    """
    :type c: int
    :rtype: bool
    """
    # ! 这个方法中设置了两个指针 low 和 high
    # ! 类似于二分法
    low = 0
    high = int(math.sqrt(c))
    while low <= high:
        value = low*low+high*high
        if value < c:
            low += 1
        elif value > c:
            high -= 1
        else:
            return True
    return False


def judgeSquareSum(self, c):  # ! 这个方法思路比较简单
    """
    :type c: int
    :rtype: bool
    """
    half = int(math.sqrt(c))

    while half >= 0:  # ! 从sqrt(c)遍历到0
        if half ** 2 == c:
            return True
        another = int(math.sqrt(c - half ** 2))

        if another ** 2 == c - half ** 2:
            return True
        half -= 1

    return False


# %%
# * Perfect Number
def checkPerfectNumber(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 1:
        return False
    half = int(math.sqrt(num))
    result = []

    while half >= 1:
        another = num / half

        if num % half == 0:
            if half is not another:
                result.append(another)
                result.append(half)
            else:
                result.append(another)

        half -= 1

    return sum(result) - num == num


# %%
# * Factorial Trailing Zeroes
def trailingZeroes(self, n):  # ! 这个解法太妙了
    """
    :type n: int
    :rtype: int
    """

    zeroCnt = 0
    while n > 0:
        n = math.floor(n/5)
        zeroCnt += n

    return zeroCnt


from functools import reduce
from fractions import Fraction


def trailingZeroes(self, n):  # ! 这个方法是可行的，但是超时了
    """
    :type n: int
    :rtype: int
    """
    if n == 0:
        return 0
    result = reduce(lambda x, y: x * y, range(1, n + 1))
    count = 0
    result = Fraction(result, 1)
    while result % 10 == 0:
        count += 1
        result = result / 10

    return count


# %%
# * Arranging Coins
def arrangeCoins(self, n):  # ! 816ms, 比较慢
    """
    :type n: int
    :rtype: int
    """
    count = 0
    if n == 0:
        return 0
    while n > 0:
        count += 1
        n = n - count

    return count if n == 0 else count - 1


def arrangeCoins(self, n):  # ! 丧心病狂,利用等差数列的性质, 64ms
    """
    :type n: int
    :rtype: int
    """
    return int(((1+8*n) ** 0.5 - 1) / 2)


# %%
def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """

    if needle in haystack:
        return haystack.index(needle)
    else:
        return -1

    # ! return haystack.index(needle) if needle in haystack else -1


# %%
# * Palindrome Linked List
import collections


def isPalindrome(self, head):  # ! 这个方法使用了额外的空间 O(n)
    """
    :type head: ListNode
    :rtype: bool
    """
    # ! 生成一个双向队列 deque
    queue = collections.deque([])
    cur = head
    # ! 让所有的元素入队列
    while cur:
        queue.append(cur)
        cur = cur.next
    while len(queue) >= 2:
        # ! 每次把队列的一头一尾的元素进行比较
        if queue.popleft().val != queue.pop().val:
            return False
    return True


def isPalindrome(self, head):  # ! 是将链表的前半段进行翻转，然后和链表的后半段进行对比
    """
    :type head: ListNode
    :rtype: bool
    """
    rev = None
    slow = fast = head
    # ! 寻找链表的中间点
    while fast and fast.next:
        fast = fast.next.next
        # ! 翻转链表的前半部分
        rev, rev.next, slow = slow, rev, slow.next
    # ! 根据链表长度的奇偶，slow指针进行相应的变化
    if fast:
        slow = slow.next
    # ! 让链表的前半段和后半段进行比较
    while rev and rev.val == slow.val:
        slow = slow.next
        rev = rev.next
    return not rev


# %%
import collections


def findPairs(self, nums, k):  # !这个逻辑没什么问题，但是代码太复杂了, 超时
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    count = 0
    if k < 0:
        return 0

    if k > 0:
        num_set = list(set(nums))
        for i in range(len(num_set)):
            if num_set[i] + k in num_set:
                count += num_set.count(num_set[i] + k)
        return count
    if k == 0:
        return sum(v > 1 for v in collections.Counter(nums).values())


def findPairs(self, nums, k):  # ! 这个就比较简洁了
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    # ! 分情况讨论
    res = 0
    c = collections.Counter(nums)  # ! 返回一个字典
    for i in c:  # ! 遍历字典的键值
        if k > 0 and i + k in c or k == 0 and c[i] > 1:
            res += 1
    return res


# %%
# * Valid Palindrome
def isPalindrome(self, s):  # ! 常规思路
    """
    :type s: str
    :rtype: bool
    """

    clean_s = [item.lower() for item in s if item.isalnum()]

    while len(clean_s) > 1:
        if clean_s.pop(0) == clean_s.pop():
            continue
        else:
            return False

    return True


import re


def isPalindrome(self, s):  # !使用正则表达式, 操作有点骚
    """
    :type s: str
    :rtype: bool
    """
    # ! 替换s字符串中非数字和字母的所有字符
    s = re.sub(r'[^A-Za-z0-9]', '', s).lower()
    return s == s[::-1]


# %%
# * Reverse Words in a String III
def reverseWords(self, s):  # ! 常规方法
    """
    :type s: str
    :rtype: str
    """
    s_arr = s.split(" ")
    result = []
    for item in s_arr:
        result.append(item[::-1])

    return " ".join(result)


# %%
# * Detect Capital
def detectCapitalUse(self, word):
    """
    :type word: str
    :rtype: bool
    """
    # ! istitle函数是用来检测字符串中的单词首字母是否写
    # ! islower(isupper)函数是用来检测字符串中所有区分大小写的符号是否是小写(大写)
    return word.isupper() or word.islower() or word.istitle()


# %%
# * Ransom Note
from collections import Counter


def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    m_dict = Counter(magazine)
    r_dict = Counter(ransomNote)

    for key, value in r_dict.items():
        if key in m_dict:
            if m_dict[key] >= value:
                continue
            else:
                return False
        else:
            return False

    return True


# %%
# * Reverse String II
def reverseStr(self, s, k):
    """
    :type s: str
    :type k: int
    :rtype: str
    """
    if len(s) == 1:
        return s

    if len(s) <= k:
        return s[::-1]

    if len(s) > k and len(s) < 2 * k:
        return s[k - 1::-1] + s[k:]

    if len(s) >= 2 * k:
        i = 0
        temp = []
        s_arr = [item for item in s]
        print(s_arr)
        while i < len(s_arr) - 1:
            temp = s_arr[i:i + k]
            temp.reverse()
            s_arr[i:i + k] = temp
            i += 2 * k

        return "".join(s_arr)


def reverseStr(self, s, k):
    """
    :type s: str
    :type k: int
    :rtype: str
    """
    i = 0
    while i*k < len(s):
        if i % 2 == 0:
            if (i+1)*k > len(s):
                s1 = s[i*k:]
                s = s[:i*k] + s1[::-1]
            else:
                s1 = s[i*k:(i+1)*k]
                s = s[:i*k] + s1[::-1] + s[(i+1)*k:]
        i += 1

    return s


# %%
# * Student Attendance Record I
def checkRecord(self, s):
    """
    :type s: str
    :rtype: bool
    """

    if s.count('A') >= 2:
        return False
    else:
        if 'LLL' not in s:
            return True
        else:
            return False

    # ! return False if s.count('A') >= 2 else True if s.count('LLL') == 0 else False  # 40ms
    # ! return s.count('A') <= 1 and 'LLL' not in s  # 36ms


# %%
# * String Compression
import re


def compress(self, chars):
    """
    :type chars: List[str]
    :rtype: int
    """
    chars[:] = re.sub(r'(?<=(.))\1+', lambda m: str(1 +
                                                    len(m.group())), ''.join(chars))
    return len(chars)


# %%
# * Number of Segments in a String
import re


def countSegments(self, s):
    """
    :type s: str
    :rtype: int
    """
    result = re.split(r'\s+', s)

    return len(result) - result.count('')
    # ! return len(s.split())


# %%
# * Judge Route Circle
from collections import Counter


def judgeCircle(self, moves):
    """
    :type moves: str
    :rtype: bool
    """
    moves_dict = Counter(moves)

    if moves_dict['U'] == moves_dict['D'] and moves_dict['R'] == moves_dict['L']:
        return True

    else:
        return False

    # ! return moves.count('L') == moves.count('R') and moves.count('U') == moves.count('D')


# %%
# * Valid Parentheses
# ! 主要使用栈stack
def isValid(self, s):  # ! 这个方法是比较巧妙，不过使用了额外的space,例如：map_dict、st
    """
    :type s: str
    :rtype: bool
    """
    map_dict = {')': '(', '}': '{', ']': '['}
    st = []
    for e in s:
        '''
          # ! 如果括号完全匹配的话，st应该为空，所以最后return not st
          # ! 只要st栈中还有剩余，就返回False
        '''
        if st and (e in map_dict and st[-1] == map_dict[e]):
            st.pop()  # ! 如果栈顶元素与当前元素匹配的话，弹出栈顶元素
        else:
            st.append(e)  # ! 如果栈顶元素与当前元素不匹配的话，压入栈中
    return not st


# %%
# * Repeated Substring Pattern
def repeatedSubstringPattern(self, s):  # ! 这种方法直接了当
    """
    :type s: str
    :rtype: bool
    """
    # ! 函数any的用法是：只要列表中有一个元素不为 False或None或''，就返回True
    # ! s[:i]从字符串头部开始取的原因是既然利用substring来reconstruct原本的字符串,必须满足字符串所有位置
    return any(s[:i] * (len(s) // i) == s for i in range(1, len(s) // 2 + 1) if len(s) % i == 0)


def repeatedSubstringPattern(self, s):  # ! 这个方法太巧妙了,一般很难能想出来
    """
    :type s: str
    :rtype: bool
    """
    # ! [1:-1]的意思是去除字符串中的第1和倒数第1位的元素
    if not s:
        return False  # ! 为空的话返回False

    return s in (2 * s)[1:-1]


# %%
# * Repeated String Match
def repeatedStringMatch(self, A, B):  # ! 这个方法思路清晰,但是太慢了
    """
    :type A: str
    :type B: str
    :rtype: int
    """
    C = ""
    for i in range(len(B)//len(A) + 3):
        if B in C:
            return i
        C += A

    return -1


def repeatedStringMatch(self, A, B):
    """
    :type A: str
    :type B: str
    :rtype: int
    """

    if B in A:
        return 1
    # ! 确认B中的所有元素都在A中，这样repeat才有意义
    if not (set(B).issubset(set(A))):
        return -1
    # ! 这一步是重点
    multiplier = -(-len(B) // len(A))  # ! 向上取整, 等同于math.ceil()
    getA = A
    getA *= multiplier
    if B in getA:
        return multiplier
    getA += A
    if B in getA:
        return multiplier + 1
    return -1


# %%
# * Hamming Distance
def hammingDistance(self, x, y):  # ! xor按位进行与操作，统计xor中'1'的个数
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    xor = x ^ y
    count = 0
    # ! _是一个没有具体意义的变量
    for _ in range(xor.bit_length()):
        count += xor & 1
        xor = xor >> 1
    return count


def hammingDistance(self, x, y):  # ! 利用count函数统计异或结果中'1'的数目
    """
    :type x: int
    :type y: int
    :rtype: int
    """

    return bin(x ^ y).count('1')


# %%
def findComplement(self, num):  # ! 这个方法相当于按位用1取扫描
    """
    :type num: int
    :rtype: int
    """
    i = 1

    while i <= num:
        i = i << 1

    return (i - 1) ^ num


def findComplement(self, num):
    """
    :type num: int
    :rtype: int
    """
    # ! bit_length函数不会获取一些没有意义的二进制位，比说：二进制数开始的许多没有意义的0
    return num ^ ((1 << num.bit_length())-1)


# %%
# *
def fizzBuzz(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    result = []

    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 != 0:
            result.append('Fizz')
            continue

        if i % 3 != 0 and i % 5 == 0:
            result.append('Buzz')
            continue

        if i % 3 == 0 and i % 5 == 0:
            result.append('FizzBuzz')
            continue

        result.append(str(i))

    return result
    # !下面这段代码是我代码的压缩版
    # ! return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]


# %%
# * Heaters
"""
遍历房屋houses，记当前房屋坐标为house：

    利用二分查找，分别找到不大于house的最大加热器坐标left，以及不小于house的最小加热器坐标right

    则当前房屋所需的最小加热器半径radius = min(house - left, right - house)

    利用radius更新最终答案ans
"""


def findRadius(self, houses, heaters):  # ! sort + 二分查找(binary search)
    """
    :type houses: List[int]
    :type heaters: List[int]
    :rtype: int
    """
    ans = 0
    heaters.sort()
    for house in houses:
        radius = 0x7FFFFFFF
        le = bisect.bisect_right(heaters, house)
        if le > 0:
            radius = min(radius, house - heaters[le - 1])
        ge = bisect.bisect_left(heaters, house)
        if ge < len(heaters):
            radius = min(radius, heaters[ge] - house)
        ans = max(ans, radius)
    return ans


# %%
# * Find Smallest Letter Greater Than Target
def nextGreatestLetter(self, letters, target):
    """
    :type letters: List[str]
    :type target: str
    :rtype: str
    """
    le = bisect.bisect_right(letters, target)

    if le == 0:
        return letters[0]

    if le > len(letters) - 1:
        return letters[0]

    return letters[le]


# %%
# * First Bad Version
# ! 折半查找
def firstBadVersion(self, n):
    """
    :type n: int
    :rtype: int
    """
    r = n-1
    le = 0
    while(le <= r):
        mid = le + (r-le)/2
        if not isBadVersion(mid):
            le = mid+1
        else:
            r = mid-1
    return le


# %%
# * Guess Number Higher or Lower
# ! 折半查找
def guessNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    low = 1
    high = n
    while True:
        mid = (low + high) / 2

        if guess(mid) == 1:
            low = mid + 1
        elif guess(mid) == 0:
            return mid
        else:
            high = mid - 1


# %%
# * Climbing Stairs
# ! 动态规划问题
def climbStairs(self, n):
    """
    :type n: int
    :rtype: int
    """
    if n <= 1:
        return 1
    res = []
    res.append(1)
    res.append(1)
    for i in range(2, n+1):
        res.append(res[-1]+res[-2])

    return res[-1]


# %%
# * Maximum Depth of Binary Tree
# ! 主要是使用递归的方式解题
def maxDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0
    leftChildHeight = self.maxDepth(root.left)  # ! 左子树递归
    rightChildHeight = self.maxDepth(root.right)  # ! 右子树递归
    return max(leftChildHeight, rightChildHeight)+1


# %%
# * Binary Number with Alternating Bits
def hasAlternatingBits(self, n):
    """
    :type n: int
    :rtype: bool
    """
    bin_num = bin(n)[2:]

    for i in range(len(bin_num) - 1):
        if bin_num[i] is bin_num[i + 1]:
            return False

    return True


# %%
# * Invert Binary Tree
def invertTree(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if root:
        root.left, root.right = self.invertTree(
            root.right), self.invertTree(root.left)

    return root


# %%
# * Prime Number of Set Bits in Binary Representation
def countPrimeSetBits(self, L, R):
    """
    :type L: int
    :type R: int
    :rtype: int
    """
    def isprime(num):
        n = 2
        while n <= int(math.sqrt(num)):
            if i_count % n == 0:
                return False
            n += 1
        else:
            return True

    count = 0
    for i in range(L, R + 1):
        i_count = bin(i).count('1')

        if i_count is 1:
            continue

        if i_count <= 3:
            count += 1
        else:
            if isprime(i_count):
                count += 1

    return count


# %%
# * Reverse Bits
# ! 用到了zfill函数，字符串右对齐
def reverseBits(self, n):
    return int(bin(n)[2:].zfill(32)[::-1], 2)


# %%
# * Base 7
def convertToBase7(self, num):  # ! 短除法
    """
    :type num: int
    :rtype: str
    """
    if num == 0:
        return '0'
    n, res = abs(num), ''
    while n:
        res = str(n % 7) + res
        n //= 7
    return res if num > 0 else '-' + res


# %%
# * Sum of Left Leaves
def sumOfLeftLeaves(self, root):  # ! 利用深度优先搜索算法
    """
    :type root: TreeNode
    :rtype: int
    """
    self.sum = 0

    def depthTral(node):  # ! 深度优先遍历
        if node:
            if node.left is not None and node.left.right is None and node.left.left is None:
                self.sum += node.left.val

            depthTral(node.left)
            depthTral(node.right)

    depthTral(root)

    return self.sum


# %%
# * Same Tree
def isSameTree(self, p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    self.result = []
    self.result1 = []

    def depthTral(node, resultArr):  # ! 深度优先搜索
        if node:
            resultArr.append(node.val)
            if node.left:
                depthTral(node.left, resultArr)
            else:
                resultArr.append('')
            if node.right:
                depthTral(node.right, resultArr)
            else:
                resultArr.append('')

    depthTral(p, self.result)
    depthTral(q, self.result1)

    return self.result == self.result1


# %%
# * Assign Cookies
def findContentChildren(self, g, s):  # ! 使用贪心算法
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """
    # ! 从小到大进行排序，可以消除原题带来的后效性
    g.sort()
    s.sort()
    # ! 双指针
    childi = 0
    cookiei = 0

    while cookiei < len(s) and childi < len(g):
        if s[cookiei] >= g[childi]:
            childi += 1
        cookiei += 1

    return childi


# %%
# *
from functools import reduce


class NumArray:  # ! 这个方法超时了

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums_arr = nums

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        result = reduce(lambda x, y: x + y, self.nums_arr[i:j + 1])

        return result


import itertools


class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.accu = [0] + list(itertools.accumulate(nums))

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.accu[j + 1] - self.accu[i]