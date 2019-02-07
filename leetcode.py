#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\leetcode.py
# Project: d:\CodeWareHouse
# Created Date: Wednesday, February 7th 2018, 4:01:41 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Monday, 19th March 2018 3:32:41 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

'''[Tips]
   # ! 双指针法(Two-Pointer)经常用在sorted的list中

'''



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


def twoSum(self, nums, target):
    # ! 这个方法比较耗时
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    for i in range(len(nums)):
        diff = target - nums[i]
        temp = nums[i + 1:]

        if diff in temp:
            return [i, temp.index(diff) + i + 1]


# %%
# * Remove Element
def removeElement(nums, val):
    # ! two-pointer 方法
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
# ! Top-Down的方法
'''
1. return specific value for null node
2. update the answer if needed                      // anwer <-- params
3. left_ans = top_down(root.left, left_params)      // left_params <-- root.val, params
4. right_ans = top_down(root.right, right_params)   // right_params <-- root.val, params
5. return the answer if needed                      // answer <-- left_ans, right_ans
'''

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
    # ! 不知道为什么之前会想这么复杂。。。。。
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


def containsDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """

    return False if len(nums) == len(set(nums)) else True


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


def twoSum(self, numbers, target):  # ! two-pointer
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    i = 0
    j = len(numbers) - 1
    res = []

    while True:
        addition = numbers[i] + numbers[j]
        if addition == target:
            res.append(i + 1)
            res.append(j + 1)
            break

        elif addition > target:
            j -= 1

        else:
            i += 1

    return res


# %%
# * Max Consecutive Ones
from collections import defaultdict


def findMaxConsecutiveOnes(self, nums):
    # ! 利用dict来存储不同区段的'1'的个数
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
            i = i + 1 # ! 如果当前的字符不是'1'而是'0'是，通过加1来更新key

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


def pivotIndex(self, nums):  # ! 11764 ms, 这个运行时间，我枯了
        """
        :type nums: List[int]
        :rtype: int
        """
        for idx in range(len(nums)):
            if sum(nums[:idx]) == sum(nums[idx + 1:]):
                return idx

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
    # ! 如果 k > 数组的长度，则取数组长度的余数作为k，例如: len = 7，k=3和k=10的结果一样
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
    # ! 只需要遍历26次, 时间复杂度为O(1)
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


def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return "".join(list(reversed(s)))


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


def intersect(self, nums1, nums2):  # ! 108ms
    # ! 这个方法太粗暴
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    res = []
    if len(nums1) > len(nums2):
        for item in nums2:
            if item in nums1:
                res.append(item)
                nums1.remove(item)

    else:
        for item in nums1:
            if item in nums2:
                res.append(item)
                nums2.remove(item)

    return res


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
# ! he Two-pointer technique, which not only applicable to Array problems but also Linked List problems as well
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
    prev = None  # ! 在这里的时候，创建一个空指针
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
    # !这个方法核心思路就是创建一个新的链表，在比较两个待merged的链表
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


def hasCycle(self, head):
    # ! 龟兔赛跑算法
    """
    :type head: ListNode
    :rtype: bool
    """
    # ! fast指针每次走两步，slow指针每次走一步
    fast = slow = head

    while fast and fast.next and fast.next.next:

        slow, fast = slow.next, fast.next.next

        # ! 如果slow == fast，说明有环
        if slow is fast:
            return True
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


def reverse(self, x):
    """
    :type x: int
    :rtype: int
    """
    if x == 0: return 0
    if x > 0:
        res = int("".join(list(reversed(str(x)))))
        return 0 if res > 2 ** 31 - 1 else res

    else:
        res = - int("".join(list(reversed(str(x)[1:]))))
        return 0 if res < - 2 ** 31 else res


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
# ! 这道题用two-pointer的方法可以做
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

def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s_arr = s.split()

        for i in range(len(s_arr)):
            s_arr[i] = "".join(list(reversed(s_arr[i])))


        return " ".join(s_arr)


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
# ! 这是一道典型的使用栈stack的题目
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

    '''
      n<=1，此时只有一种。

　　   n>1时，对于每一个台阶i，要到达台阶，最后一步都有两种方法，从i-1迈一步，或从i-2迈两步。

　　   也就是说到达台阶i的方法数=达台阶i-1的方法数+达台阶i-2的方法数。所以该问题是个DP问题。
    '''

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
# * Range Sum Query - Immutable
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


class NumArray:  # ! 使用itertools的accumulate方法

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


# %%
def isPalindrome(self, x):  # ! 288ms, 时间有点长
    """
    :type x: int
    :rtype: bool
    """
    x = str(x)
    # if len(x) is 1: return True
    i = 0
    j = len(x) - 1
    while i <= len(x) / 2:
        if x[i] is not x[j]:
            return False
        i += 1
        j -= 1

    else:
        return True


# %%
def findRelativeRanks(self, nums):
    """
    :type nums: List[int]
    :rtype: List[str]
    """
    new_num = nums[:]
    num_dict = {}
    for i, v in enumerate(sorted(nums, reverse=True)):
        num_dict[v] = i + 1

    return ["Gold Medal" * (num_dict[item] is 1) or "Silver Medal" * (num_dict[item] is 2) or "Bronze Medal" * (num_dict[item] is 3) or str(num_dict[item]) for item in new_num]


# %%
# * Power of Four
def isPowerOfFour(self, num):
    """
    :type num: int
    :rtype: bool
    """
    while num % 4 == 0 and num is not 0:
        num = num / 4
    else:
        return True if int(num) is 1 else False

    # ! 下面的方法比较巧妙，利用了题设中的
    # ! return num != 0 and num &(num-1) == 0 and num & 1431655765== num


# %%
# * Minimum Depth of Binary Tree
def minDepth(self, root):  # ! 56ms 使用广度优先搜索
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0

    depth = 1

    queue = []
    queue.append(root)
    count = 0
    lastnum = 1

    while queue:
        node = queue.pop(0)
        print(node.val)

        if not node.left and not node.right:
            return depth
        lastnum -= 1

        if node.left:
            queue.append(node.left)
            count += 1
        if node.right:
            queue.append(node.right)
            count += 1

        if lastnum is 0:
            lastnum = count
            count = 0
            depth += 1


# %%
# * Min Stack
class MinStack:  # ! 768ms，比较慢

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return min(self.stack)


class MinStack:  # ! 这个方法快只需要64ms

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)
        if not self.stack2 or self.stack2[-1] >= x:
            self.stack2.append(x)

    def pop(self):
        tmp = self.stack1[-1]
        self.stack1.pop()
        if self.stack2 and tmp == self.stack2[-1]:
            self.stack2.pop()
        return tmp

    def top(self):
        return self.stack1[-1]

    def getMin(self):
        return self.stack2[-1]


# %%
# * Binary Watch
def readBinaryWatch(self, num):  # ! 44ms，进行双重循环，并统计1的数量，没什么太多的技巧性
    """
    :type num: int
    :rtype: List[str]
    """
    return ['%d:%02d' % (h, m)
            for h in range(12) for m in range(60)
            if (bin(h) + bin(m)).count('1') == num]


# %%
# * Binary Tree Paths
# ! 这道题的基本思路是采用深度优先搜索算法
def binaryTreePaths(self, root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    if not root.left and not root.right:
        return [str(root.val)]

    return [str(root.val) + '->' + i for i in self.binaryTreePaths(root.left)] + [str(root.val) + '->' + i for i in self.binaryTreePaths(root.right)]


def binaryTreePaths(self, root):  # ! 这个思路我想到了，但是实现的方式有点问题
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    res = []
    self.dfs(root, "", res)
    return res

    def dfs(self, root, ls, res):
        if not root.left and not root.right:
            res.append(ls+str(root.val))
        if root.left:
            self.dfs(root.left, ls+str(root.val)+"->", res)
        if root.right:
            self.dfs(root.right, ls+str(root.val)+"->", res)


# %%
# * Second Minimum Node In a Binary Tree
# ! 使用深度优先搜索遍历每个结点
def findSecondMinimumValue(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.res = []
    if not root:
        return -1

    def dfs(node):
        if node:
            self.res.append(node.val)

            if node.left:
                dfs(node.left)

            if node.right:
                dfs(node.right)

    dfs(root)
    res_set = list(set(self.res))
    res_set.sort()

    return -1 if len(res_set) is 1 else res_set[1]


# %%
# * Implement Queue using Stacks
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.stack.pop(0)

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.stack[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return True if not self.stack else False


# %%
# * Convert a Number to Hexadecimal
def toHex(self, num):  # ! 这个方法是作弊的，我只是想测试一下时间，40ms
    """
    :type num: int
    :rtype: str
    """
    if num < 0:
        num += 2 ** 32

    return hex(num)[2:]


def toHex(self, num):  # ! 这个是常规方法， 37ms
    """
    :type num: int
    :rtype: str
    """
    if num is 0:
        return '0'
    if num < 0:
        num += 2 ** 32

    dict_hex = ['0', '1', '2', '3', '4', '5', '6',
                '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    res = ''

    # ! 短除法，用于解决进制转换问题
    while num:

        res += dict_hex[num % 16]
        num = num // 16

    else:
        return res[::-1]


# %%
# * Implement Stack using Queues
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.stack.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.stack[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return True if not self.stack else False


# %%
# *  Two Sum IV - Input is a BST
def findTarget(self, root, k):  # ! 736ms，太慢了，使用深度优先搜索
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    if not root:
        return False
    self.res = []

    def DSTFind(node):
        if node:
            if k - node.val in self.res:
                self.res.append('True')
                return

            self.res.append(node.val)

            if node.left:
                DSTFind(node.left)

            if node.right:
                DSTFind(node.right)

    DSTFind(root)

    return True if 'True' in self.res else False


def findTarget(self, root, k):  # ! 104ms，利用广度优先搜索
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    if not root:
        return False
    res = []
    bfs = [root]

    while bfs:
        node = bfs.pop(0)
        if k - node.val in res:
            return True

        res.append(node.val)

        if node.left:
            bfs.append(node.left)

        if node.right:
            bfs.append(node.right)

    return False


# %%
# * Construct the Rectangle
def constructRectangle(self, area):
    """
    :type area: int
    :rtype: List[int]
    """
    # ! int函数会自动向下取整
    mid = int(math.sqrt(area))
    while mid > 0:
        if area % mid == 0:
            # ! [L, W]
            return [int(area / mid), int(mid)]
        mid -= 1


# %%
# * Longest Common Prefix
def longestCommonPrefix(self, strs):  # ! 44ms
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ""

    length = 0
    # ! 这个方法中主要是zip这方法用得十分巧妙
    # ! 每一个item就是字符串数组中每个字符串的字符组成一个tuple，例如：('','','')
    for item in list(zip(*strs)):

        if len(set(item)) > 1:
            return strs[0][:length]

        length += 1

    return strs[0][:length]


# %%
# * Average of Levels in Binary Tree
def averageOfLevels(self, root):  # ! 利用广度优先搜索，64ms
    """
    :type root: TreeNode
    :rtype: List[float]
    """
    if not root:
        return []
    bfs = [root]
    res = []
    tempres = []
    lastnum = 1
    count = 0

    while bfs:
        node = bfs.pop(0)
        tempres.append(node.val)
        lastnum -= 1

        if node.left:
            bfs.append(node.left)
            count += 1

        if node.right:
            bfs.append(node.right)
            count += 1

        if lastnum is 0:
            lastnum = count
            count = 0
            res.append(sum(tempres) / len(tempres))
            tempres = []

    return res


# %%
# * Longest Harmonious Subsequence
# ! 这道题其实就是再统计x和x + 1的总和最大的那组数据的长度
from collections import Counter


def findLHS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    count = collections.Counter(nums)
    ans = 0
    for x in count:
        if x + 1 in count:
            ans = max(ans, count[x] + count[x+1])
    return ans


# %%
# * Daily Temperatures
def dailyTemperatures(self, temperatures):
    """
    :type temperatures: List[int]
    :rtype: List[int]
    """
    # ! O(n ** 2)的方法是肯定可以做的，下面这个方法思路更加巧妙，时间复杂度只有O(n)

    res = [0 for i in range(len(temperatures))]
    stack = []
    for i, tem in enumerate(temperatures):

        # ! 当当前元素大于stack中最后一个元素时，修改res中相应位置的值，否则跳出循环
        while len(stack) and tem > stack[-1][1]:
            fi, ft = stack.pop()
            res[fi] = i - fi
        stack.append((i, tem))
    return res


# %%
# * Backspace String Compare
def backspaceCompare(self, S, T):
    """
    :type S: str
    :type T: str
    :rtype: bool
    """

    def backspace_op(string):

        stack = []

        for i, item in enumerate(string):

            if item == '#' and len(stack) > 0:
                stack.pop()

            if item != '#':
                stack.append(item)

        return "".join(stack)

    S_stack = backspace_op(S)
    T_stack = backspace_op(T)

    return S_stack == T_stack


# %%
# * Reverse Vowels of a String
def reverseVowels(self, s):
    """
    :type s: str
    :rtype: str
    """
    ss = list(s)
    aeiou = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']  # ! 元音字符列表
    n = len(s)
    i = 0       # ! 头指针
    j = n - 1     # ! 尾指针

    while i < j:
        if ss[i] not in aeiou:  # ! 如果头不是是元音字符，向后移动，跳过循环
            i += 1
            continue
        if ss[j] not in aeiou:  # ! 如果尾是元音字符，向前移动，跳过循环
            j -= 1
            continue
        if (i < j):     # ! 如果头指针和尾指针同时指向元音字符，交换
            ss[i], ss[j] = ss[j], ss[i]
            i += 1
            j -= 1
    return "".join(ss)


# %%
# * 1-bit and 2-bit Characters
def isOneBitCharacter(self, bits):
    """
    :type bits: List[int]
    :rtype: bool
    """
    length = len(bits)
    if length == 1:
        return True

    if length == 2:
        if bits[0] == 1:
            return False
        else:
            return True

    i = 0
    while i < length - 1:
        if bits[i] == 0:
            i += 1
        else:
            i += 2

    return True if i == length - 1 else False


# %%
# * Buddy Strings
from collections import Counter


def buddyStrings(self, A, B):
    """
    :type A: str
    :type B: str
    :rtype: bool
    """

    zip_ab = zip(A, B)  # ! 要会活用zip函数，zip函数相当于多一条思路

    count_a = Counter(A)
    count_b = Counter(B)

    count_num = sum([x != y for x, y in zip_ab])

    # ! 这是这道题的精髓所在
    return count_num <= 2 and count_a == count_b if A != B else len(set(A)) < len(A)


# %%
# * Most Common Word
def mostCommonWord(self, paragraph, banned):
    """
    :type paragraph: str
    :type banned: List[str]
    :rtype: str
    """
    ban = set(banned)
    # ! 将paragraph中所有非字母的符号转化为空格
    words = re.sub(r'[^a-zA-Z]', ' ', paragraph).lower().split()
    # ! 除去banned数组中的值，再利用Counter获取最高频率的词
    return Counter(w for w in words if w not in ban).most_common(1)[0][0]


# %%
# * Search in Rotated Sorted Array
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    try:
        return nums.index(target)
    except:
        return -1


# %%
# * Baseball Game
def calPoints(self, ops):
    """
    :type ops: List[str]
    :rtype: int
    """
    ops_stack = []

    while ops:
        op = ops.pop(0)

        if op.isdigit():
            ops_stack.append(int(op))
            continue

        if '-' in op:
            ops_stack.append(int(op))
            continue

        if op == 'C':
            ops_stack.pop()
            continue

        if op == 'D':
            ops_stack.append(ops_stack[-1] * 2)
            continue

        if op == '+':
            ops_stack.append(ops_stack[-1]  + ops_stack[-2])

    return sum(ops_stack)


# %%
# * Implement Magic Dictionary


class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.default_dict = collections.defaultdict(set)

    def buildDict(self, dict):
        """
        Build a dictionary through a list of words
        :type dict: List[str]
        :rtype: void
        """
        for item in dict:
            self.default_dict[len(item)].add(item)

    def search(self, word):
        """
        Returns if there is any word in the trie that equals to the given word after modifying exactly one character
        :type word: str
        :rtype: bool
        """
        k = len(word)

        if k in self.default_dict:
            for idx, item in enumerate(self.default_dict[k]):
                if item == word:
                    continue
                res = [1 for i in range(0, k) if item[i] != word[i]]

                if sum(res) == 1:
                    return True

            return False
        else:
            return False

# %%
# * Best Time to Buy and Sell Stock II
def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if prices == []:
            return 0

        profits = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        if prices.index(max(prices)) > prices.index(min(prices)):
            range_profits = max(prices) - min(prices)
        else:
            range_profits = 0
        all_profits = 0

        for item in profits:
            if item > 0:
                all_profits += item

        if all_profits == 0:
            return 0


        return all_profits if all_profits > range_profits else range_profits

# %%
# * Magic Squares In Grid
def numMagicSquaresInside(self, grid):
    # ! 这道题极其的无聊
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    count = 0
    for i in range(len(grid) - 2):
        for j in range(len(grid[i]) - 2):
            test_arr = []
            test_arr.append(grid[i][j])
            test_arr.append(grid[i][j + 1])
            test_arr.append(grid[i][j + 2])
            test_arr.append(grid[i + 1][j])
            test_arr.append(grid[i + 1][j + 1])
            test_arr.append(grid[i + 1][j + 2])
            test_arr.append(grid[i + 2][j])
            test_arr.append(grid[i + 2][j + 1])
            test_arr.append(grid[i + 2][j + 2])
            if len(set(test_arr)) < 9:
                continue
            test_temp = 0
            for item in test_arr:
                if item >= 1 and item <= 9:
                    test_temp += 1

            if test_temp != 9:
                continue
            magic_sum = grid[i][j] + grid[i][j + 1] + grid[i][j + 2]
            magic_next = grid[i + 1][j] + grid[i + 1][j + 1] + grid[i + 1][j + 2]
            magic_last = grid[i + 2][j] + grid[i + 2][j + 1] + grid[i + 2][j + 2]
            magic_diagonal = grid[i][j] + grid[i + 1][j + 1] + grid[i + 2][j + 2]
            magic_re_diagonal = grid[i][j + 2] + grid[i + 1][j + 1] + grid[i + 2][j]
            magic_col = grid[i][j] + grid[i + 1][j] + grid[i + 2][j]
            magic_col_next = grid[i][j + 1] + grid[i + 1][j + 1] + grid[i + 2][j + 1]
            magic_col_last = grid[i][j +2] + grid[i + 1][j + 2] + grid[i + 2][j + 2]

            if magic_sum == magic_next and magic_sum == magic_last and magic_sum == magic_diagonal and magic_sum == magic_re_diagonal and magic_sum and magic_sum == magic_col and magic_sum == magic_col_next and magic_sum == magic_col_last:
                count += 1


    return count

# %%
# * Reveal Cards In Increasing Order
def deckRevealedIncreasing(self, deck):
    # ! 按照题目给的思路倒过来求解
    """
    :type deck: List[int]
    :rtype: List[int]
    """
    if not deck:
        return []

    deck = sorted(deck, reverse=True)
    reorder = [deck[0]]

    for i in range(1, len(deck)):
        reorder.insert(0, reorder[-1])
        temp = reorder.pop(-1)
        reorder.insert(0, deck[i])


    return reorder

# %%
# * Subarray Sum Equals K
# ! perfix Sum
# ! As is typical with problems involving subarrays, we use prefix sums to add each subarray
def subarraySum(self, nums, k):
    # ! 暴力求解，无法通过，超时
    # ! O(n^2 + n)
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    hash_dict = {}
    acc = 0
    count = 0

    for item in enumerate(nums):
        acc += item[1]
        hash_dict[item[0]] = acc


    hash_arr = list(hash_dict.values())

    for i in range(len(hash_arr)):
        for j in range(i, len(hash_arr)):
            diff = hash_arr[j] - hash_arr[i] + nums[i]

            if diff == k:
                count += 1

    return count


def subarraySum(self, nums, k):
    # ! 这样也是暴力求解，O(n^2)
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    acc = 0
    count = 0

    for i in range(len(nums)):
        acc = nums[i]
        if acc == k:
            count += 1
        for j in range(i + 1, len(nums)):
            acc += nums[j]

            if acc == k:
                count += 1

    return count


def subarraySum(self, nums, k):
    # * 最佳方案
    # ! 建立哈希表的目的是为了让我们可以快速的查找sum-k是否存在，即是否有连续子数组的和为sum-k
    # ! 如果sum-k存在于数组中的话，表示k肯定能在数组总找到
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    res = 0
    dic = {0: 1} # ! 初始化
    summ = 0
    for n in nums:
        summ += n
        diff = summ - k
        if diff in dic:
            res += dic[diff]
        if summ in dic:
            dic[summ] += 1
        else:
            dic[summ] = 1
    return res


# %%
# * Combination Sum
def combinationSum(self, candidates, target):
    # ! 这道题需要用到递归和回溯法
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    temp = []
    res = []

    # ! 回溯法用深度优先搜索(DFS)来实现
    def bf_dfs(target, k, candidates, temp, res):

        # ! 递归退出条件
        if target == 0:
            res.append(temp[:])
            return

        # ! 循环递归
        for i in range(k, len(candidates)):
            # ! 直接跳过相同元素
            if i != 0 and candidates[i] == candidates[i - 1]:
                continue

            # ! 计算差距或者差异
            diff = target - candidates[i]

            if diff >= 0:
                temp.append(candidates[i])
                # ! 递归，深度优先搜索
                bf_dfs(diff, i, candidates, temp, res)
                del temp[-1]

        return

    bf_dfs(target, 0, candidates, temp, res)

    return res


# %%
# * Combination Sum II
def combinationSum2(self, candidates, target):
    # ! 1、在同一层递归树中，如果某元素已经处理并进入下一层递归，那么与该元素相同的值就应该跳过。否则将出现重复
    # ! 2、相同元素第一个进入下一层递归，而不是任意一个
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    temp = []
    res = []
    final_res = []

    candidates = sorted(candidates) # ! 先按从小到大的顺序排序

    # ! 回溯法用深度优先搜索(DFS)来实现
    def bf_dfs(target, k, candidates, temp, res):

        # ! 递归退出条件
        if target == 0:
            res.append(temp[:])
            return

        # ! 循环递归
        for i in range(k, len(candidates)):
            # !因为有重复元素，所以保证相同元素第一个进入下一层递归
            if i != k and candidates[i] == candidates[i - 1]:
                continue

            # ! 计算差距或者差异
            diff = target - candidates[i]

            if diff >= 0:
                temp.append(candidates[i])
                # ! 递归，深度优先搜索
                bf_dfs(diff, i + 1, candidates, temp, res)
                del temp[-1]

        return

    bf_dfs(target, 0, candidates, temp, res)

    return res


# %%
# * Letter Case Permutation
def letterCasePermutation(self, S):
    # ! 超时
    """
    :type S: str
    :rtype: List[str]
    """
    if S.isdigit():
        return [S]

    if S == "":
        return [""]

    res = [S]

    str_arr = list(S)

    def bf_dfs(str_init, cha):
        temp = copy.deepcopy(str_init)

        if temp[cha].isalpha():
            temp[cha] = temp[cha].lower()
            str_init[cha] = str_init[cha].upper()

            res.append("".join(str_init))
            res.append("".join(temp))


        for i in range(cha + 1, len(temp)):
            bf_dfs(temp, i)
            bf_dfs(str_init, i)

        return

    bf_dfs(str_arr, 0)

    return list(set(res))


def letterCasePermutation(self, S):
    # ! 283ms, 踩线过
    # ! 使用深度优先搜索
    """
    :type S: str
    :rtype: List[str]
    """
    if S == "":
        return [""]

    res = [S]

    str_arr = list(S)

    def bf_dfs(str_init, cha):

        if cha == len(str_init):
            return

        if str_init[cha].isalpha():
            temp = copy.deepcopy(str_init)
            temp[cha] = temp[cha].lower()
            str_init[cha] = str_init[cha].upper()

            res.append("".join(str_init))
            res.append("".join(temp))
            bf_dfs(temp, cha + 1)
            bf_dfs(str_init, cha + 1)
        else:
            bf_dfs(str_init, cha + 1)


    bf_dfs(str_arr, 0)

    return list(set(res))


# %%
# * Powerful Integers
def powerfulIntegers(self, x, y, bound):
    """
    :type x: int
    :type y: int
    :type bound: int
    :rtype: List[int]
    """
    if bound == 0:
        return []

    if x == 1:
        x_bound = 0
    else:
        x_bound = math.ceil(math.log(bound, x))

    if y == 1:
        y_bound = 0
    else:
        y_bound = math.ceil(math.log(bound, y))

    res = []

    for i in range(x_bound + 1):
        for j in range(y_bound + 1):
            acc = int(math.pow(x, i) + math.pow(y, j))

            if acc <= bound and acc not in res:
                res.append(acc)


    return res


# %%
# * Pancake Sorting
def pancakeSort(self, A):
    # ! 这个方法实际上是在每轮循环中寻找最大的那个数，使其在正确的位置
    """
    :type A: List[int]
    :rtype: List[int]
    """
    bucket = sorted(A)
    ans = []
    for k in range(len(A),0,-1):
        i = A.index(bucket.pop())+1
        ans += [i, k]
        A = A[i:k][::-1] + A[:i]  + A[k:]
        print(A)
    return ans


# %%
# * Construct Binary Tree from Preorder and Inorder Traversal
def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    # ! 中序遍历序列
    inorder_dict = {item[1]:item[0] for item in enumerate(inorder)} # ! 左<中<右
    head = None
    stack_tree = []

    for val in preorder: # ! 中<左<右，前序遍历序列
        if not head:
            head = TreeNode(val)
            stack_tree.append(head)
        else:
            node = TreeNode(val)
            if inorder_dict[val] < inorder_dict[stack_tree[-1].val]:
                stack_tree[-1].left = node
            else:
                # ! 取位于当前节点的左边的那个节点作为其根节点，即u
                while stack_tree and inorder_dict[val] > inorder_dict[stack_tree[-1].val]:
                    u = stack_tree.pop()
                u.right = node

            stack_tree.append(node)


    return head


# %%
# * Flip String to Monotone Increasing
def minFlipsMonoIncr(self, S):
    # ! 这个方法超时了，不过这个方法就是暴力求解，寻找字符串的最佳切分点
    # ! 不过for循环中的count方法太耗时，不要使用
    """
    :type S: str
    :rtype: int
    """

    count0 = S.count('0')
    count1 = S.count('1')
    minimum_count = min(count0, count1)

    for i in range(1, len(S)):
        count = S[:i].count('1') + S[i:].count('0')

        if count < minimum_count:
            minimum_count = count

    return minimum_count


def minFlipsMonoIncr(self, S):
    # ! 84ms
    # ! 这道题可以转化为寻找字符串最佳切分点的问题来解
    """
    :type S: str
    :rtype: int
    """
    count0 = S.count('0')
    count1 = S.count('1')
    minimum_count = min(count0, count1)
    left_1, right_0 = 0, count0

    # ! 主要是将count部分优化了一下
    # ! 因为需要将S字符串从左往右遍历一遍，所以直接在遍历的过程中，计算出现在左边1和出现在右边0的数量
    for i in range(0, len(S)):
        if S[i] == '1':
            left_1 += 1
        else:
            right_0 -= 1

        count = left_1 + right_0

        if count < minimum_count:
            minimum_count = count

    return minimum_count


# %%
# * Max Chunks To Make Sorted
# ! 当前数字所在的块至少要到达坐标为当前数字大小的地方，比如数字4所在的块至少要包括i=4的那个位置
# ! 对例子的分析和思考很重要，仔细理解题目
# ! 这个思路确实是最好最快的思路
# ! 递归也可以实现，但是时间成本比较高，坑比较多，不推荐
def maxChunksToSorted(self, arr):
    """
    :type arr: List[int]
    :rtype: int
    """
    count, current_max = 0, float('-inf')
    for i, a in enumerate(arr):
        current_max = max(current_max, a)
        if i == current_max:
            count += 1

    return count


# %%
# * Custom Sort String
def customSortString(self, S, T):
    # ! 126ms, 踩线过
    # ! 这道题题意很明确, 比较容易理解
    # ! 暴力求解, O(n^2)
    """
    :type S: str
    :type T: str
    :rtype: str
    """
    s_dict = {item[1]:item[0] for item in enumerate(S)}

    T_arr = list(T)

    # ! 冒泡排序, 时间复杂度比较高
    for i in range(len(T_arr)):
        for j in range(i+1, len(T_arr)):

            if T_arr[i] in s_dict and T_arr[j] in s_dict:
                if s_dict[T_arr[i]] > s_dict[T_arr[j]]:
                    T_arr[i], T_arr[j] = T_arr[j], T_arr[i]


    return "".join(T_arr)


def customSortString(self, S, T):
    # ! 56ms，构造一个新的字符串(即排序后的字符串)
    # ! 时间复杂度低
    """
    :type S: str
    :type T: str
    :rtype: str
    """
    # ! 这个count函数用得很精髓
    result = ""
    for char in S:
        if char in T:
            count = T.count(char) # !
            result += char * count
            T = T.replace(char,'')
    return result + T



# %%
# * Fibonacci Number
# ! 这是一道垃圾题目, 这个有点弱智(当作回顾)
def fib(self, N):
    """
    :type N: int
    :rtype: int
    """

    def cal_fib(n):
        if n == 0:
            return 0

        if n == 1:
            return 1


        return cal_fib(n - 1) + cal_fib(n - 2)


    return cal_fib(N)


# %%
# * Diagonal Traverse
def findDiagonalOrder(self, matrix):  # ! 168ms
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    if matrix == []:
        return matrix

    m = len(matrix)
    n = len(matrix[0])

    if m == 1:
        return matrix[0]

    if n == 1:
        return [item[0] for item in matrix]
    i, j = 0, 1
    flag = 0 # ! 这帮助辨别方向

    res = [matrix[0][0]]

    while True:

        res.append(matrix[i][j])

        if i == m - 1 and j == n - 1:
            break

        if j == 0 and i < m - 1 and flag == 0:
            i += 1
            flag = 1
            continue

        if j < n - 1 and i == 0 and flag == 1:
            flag = 0
            j += 1
            continue

        if i == m - 1 and j < n - 1 and flag == 0:
            flag = 1
            j += 1
            continue

        if j == n - 1 and i < m - 1 and flag == 1:
            flag = 0
            i += 1
            continue

        if flag == 0:
            i += 1
            j -= 1
            continue

        if flag == 1:
            i -= 1
            j += 1
            continue

    return res


# %%
# * Spiral Matrix
def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    result = []
    if matrix == []:
        return result

    # ! 将这道题从左，右，上，下四个方位拆解，进行作答
    # ! 其实就是将螺旋的过程进行拆分
    # ! 将一个复杂的过程，进行拆分实现，有助于答题以及理清楚逻辑
    left, right, top, bottom = 0, len(matrix[0]) - 1, 0, len(matrix) - 1

    while left <= right and top <= bottom:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        for i in range(top + 1, bottom):
            result.append(matrix[i][right])
        for j in reversed(range(left, right + 1)):  # ! reversed是个迭代器
            if top < bottom:
                result.append(matrix[bottom][j])
        for i in reversed(range(top + 1, bottom)):
            if left < right:
                result.append(matrix[i][left])
        left, right, top, bottom = left + 1, right - 1, top + 1, bottom - 1

    return result


# %%
# * Minimum Size Subarray Sum
'''
我们需要定义两个指针left和right，分别记录子数组的左右的边界位置，
然后我们让right向右移，直到子数组和大于等于给定值或者right达到数组末尾，此时我们更新最短距离，
并且将left像右移一位，然后再sum中减去移去的值，然后重复上面的步骤
'''
def minSubArrayLen(self, s, nums):
    # ! 从two-pointer的角度出发进行思考解法
    """
    :type s: int
    :type nums: List[int]
    :rtype: int
    """
    if nums == []: return 0
    if max(nums) >= s: return 1
    left, right = 0, 0
    acc = 0
    length_num = len(nums)
    res = float('inf')


    for i in range(len(nums)):
        acc += nums[i]

        while left <= i and acc >= s:
            res = min(res, i - left + 1)
            acc -= nums[left]
            left += 1



    return 0 if res == float('inf') else res



# %%
# * Remove Duplicates from Sorted Array
def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) == 0:
        return 0
    j = 0
    len_n = len(nums)
    for i in range(len_n):

        if nums[j] != nums[i]:
            nums[j + 1] = nums[i]
            j += 1

    return j + 1


# %%
# * Valid Sudoku
def isValidSudoku(self, board):
    """
    :type board: List[List[str]]
    :rtype: bool
    """
    # ! 创建字典保存board中的信息是很巧妙的想法
    col_dict = [{} for i in range(9)]
    row_dict = [{} for i in range(9)]
    box_dict = [{} for i in range(9)]

    for i in range(9):
        for j in range(9):
            num = board[i][j]

            if num != '.':
                num = int(num)

                # ! 这个是这道题的关键步骤之一
                box_index = (i // 3) * 3 + j // 3

                row_dict[i][num] = row_dict[i].get(num, 0) + 1
                col_dict[j][num] = col_dict[j].get(num, 0) + 1
                box_dict[box_index][num] = box_dict[box_index].get(
                    num, 0) + 1

                if row_dict[i][num] > 1 or col_dict[j][num] > 1 or box_dict[box_index][num] > 1:
                    return False

        return True


# %%
# * Rotate Image
def rotate(self, matrix):
    # ! 可以通过观察exmaples来寻找解题思路
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix)

    # ! 先沿对角线交换元素
    for i in range(n):
        for j in range(i): # ! i
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


    # ! 再按照相应的两列进行交换元素
    for i in range(n):
        for j in range(n // 2):
            matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]


# %%
# * K Closest Points to Origin
def kClosest(self, points, K):
    """
    :type points: List[List[int]]
    :type K: int
    :rtype: List[List[int]]
    """
    res = []
    def eclidean_distance(x, y):

        return math.sqrt(x ** 2 + y ** 2)


    distances_points = sorted({item[0]:eclidean_distance(item[1][0], item[1][1]) for item in enumerate(points)}.items(), key=lambda item: item[1])



    for i in range(0, K):
        res.append(points[distances_points[i][0]])

    return res


# %%
# *
def largestPerimeter(self, A):
    # ! 使用遍历的方法无法通过，会超时
    # ! 我思考这个题的时候思想有点固化，不一定要完全列举出所有combinations
    # ! 这种想法是导致超时的直接原因
    """
    :type A: List[int]
    :rtype: int
    """
    max_length = 0

    def isTriangle(x,y,z):
        nums = sorted([x,y,z])

        if nums[0] + nums[1] > nums[2] and nums[2] - nums[0] < nums[1]:
            return True
        else:
            return False

    for item in list(combinations(A, 3)):
        temp = item[0] + item[1] + item[2]
        if isTriangle(item[0], item[1], item[2]) and max_length < temp:
            max_length = temp



    return max_length



def largestPerimeter(self, A):
    """
    :type A: List[int]
    :rtype: int
    """

    # ! sorted用得很精髓
    A = sorted(A, reverse=True)

    for i in range(0, len(A)-2):
        if A[i+1] + A[i+2] > A[i]:
            return A[i] + A[i+1] + A[i+2]

    return 0



# %%
# * Subarray Sums Divisible by K
# ! (a+b+c+d....)% K = (a%K + b%K +....)%K
# ! so if their prefix sum remainder subtract to zero, it is a valid subarray sum divisible to K
def subarraysDivByK(self, A, K):
    # ! 超时
    # ! 最为弱智的一种方法
    """
    :type A: List[int]
    :type K: int
    :rtype: int
    """

    count = 0
    acc = 0

    for i in range(len(A)):
        acc += A[i]
        if acc % K == 0:
            count += 1
        for j in range(i + 1, len(A)):
            acc += A[j]
            if acc % K == 0:
                count += 1

        acc = 0



    return count


def subarraysDivByK(self, A, K):
    # !利用余数的性质, 以及数组前缀和
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        hashMap = {}
        cnt = 0
        add = 0
        for a in A:
            add += a
            mod = add % K

            if mod == 0:
                cnt += 1
            if mod in hashMap:
                cnt += hashMap[mod]
                hashMap[mod] += 1
            else:
                hashMap[mod] = 1

        return cnt


# %%
# * Count and Say
def countAndSay(self, n):
    # ! 这题的考点在于迭代生成字符串
    """
    :type n: int
    :rtype: str
    """
    b='1'   # 将第一行的1换成字符类型，便于下一行的读出
    for i in range (n-1):    # (n-1)是因为第一行不需要处理，直接可以读出
        a, c, count = b[0], '', 0   # a用来读取上一行的第一个字符，c用来存放读出的内容(char)，count用来统计
        for j in b:
            if a == j:
                count += 1
            else:
                # ! 在遇到新的字符是，先将之前的结果存入最终的结果里
                c += str(count) + a   # 注意一定要将count转换为字符型，否则两个数就会相加（变成数学公式）。
                a = j
                count = 1
        c += str(count) + a
        b = c
    return b


# %%
# * Remove Nth Node From End of List
def removeNthFromEnd(self, head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    len_count = 0
    count = 0

    cur = res = head

    while cur:
        len_count += 1
        cur = cur.next

    if len_count == 1:
        head = None
        return head

    if n == 1:  # ! 处理尾删除情况
        while head:
            count += 1
            if count == len_count - 1:
                head.next = None
                return res
            head = head.next



    while head:
        # !处理头删除情况
        if len_count - count == n and count == 0:
            head = head.next
            return head

        if len_count - count == n:
            head.val = head.next.val
            head.next = head.next.next


        count += 1

        head = head.next

    return res


# %%
# * Binary Tree Level Order Traversal
def levelOrder(self, root):
    # ! 使用广度优先搜索进行解题
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root: return []
    res = []
    Bst = [root]
    lastNum = 1
    count = 0
    temp = []

    while Bst:
        node = Bst.pop(0)
        lastNum -= 1
        temp.append(node.val)


        if node.left:
            count += 1
            Bst.append(node.left)

        if node.right:
            count += 1
            Bst.append(node.right)

        if lastNum == 0:
            res.append(temp)
            temp = []
            lastNum = count
            count = 0

    return res


# %%
# * Symmetric Tree
def isSymmetric(self, root):
    # ! 利用广度优先搜索，超时
    """
    :type root: TreeNode
    :rtype: bool
    """
    def compare_sym(arr):
        if len(arr) == 1:
            return True
        j = int(len(arr) / 2)
        i = j - 1

        while i >= 0 and j <= len(arr) - 1:
            if arr[i] == arr[j]:
                i -= 1
                j += 1
                continue
            else:
                return False

        return True

    if not root: return True

    temp = []
    bfs = [root]
    lastNum = 1
    count = 0

    while bfs:
        node = bfs.pop(0)
        lastNum -= 1
        temp.append(node.val)

        if node.left:
            count += 1
            bfs.append(node.left)
        else:
            count += 1
            bfs.append(TreeNode(0))

        if node.right:
            count += 1
            bfs.append(node.right)
        else:
            count += 1
            bfs.append(TreeNode(0))


        if lastNum == 0:
            if list(set(temp)) == [0] and len(temp) > 1:
                break
            lastNum = count
            count = 0
            if compare_sym(temp):
                temp = []
            else:
                return False

    return True


def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    if not root:
        return True

    temp = []
    # ! 考虑的时候，思维有点固化，一个list不行，可以两个
    bfs_left = [root.left]
    bfs_right = [root.right]

    while bfs_left and bfs_right:
        # ! 分左右子树进行迭代
        node_left = bfs_left.pop(0)
        node_right = bfs_right.pop(0)

        if not node_left and not node_right:
            continue

        if (not node_left and node_right) or (node_left and not node_right):
            return False

        if node_left.val != node_right.val:
            return False

        bfs_left.append(node_left.left)
        bfs_left.append(node_left.right)
        # ! 注意bfs_right是反方向添加node，这样便于比较
        bfs_right.append(node_right.right)
        bfs_right.append(node_right.left)

    return True


# %%
# * Shuffle an Array
class Solution:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.num_set = nums
        self.orgin_set = self.num_set.copy()

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.orgin_set


    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        random.shuffle(self.num_set)
        return self.num_set


# %%
# * Number of 1 Bits
def hammingWeight(self, n):
    """
    :type n: int
    :rtype: int
    """
    return bin(n)[2:].count('1')


# %%
# * Reverse Bits
def reverseBits(self, n):
    # ! @param n, an integer
    # ! @return an integer
    bin_n = bin(n)[2:]

    if len(bin_n) < 32:
        bin_n = '0' * (32 - len(bin_n)) + bin_n

    return int(bin_n[::-1], 2)


# %%
# * House Robber
def rob(self, nums):
    # ! 错误解法，只考虑到一步的动态规划
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)

    if n == 0: return 0

    if n == 1: return nums[0]

    res = [nums[0], nums[1]]

    for i in range(2, n):
        res.append(res[-2] + nums[i])


    return max(res[-2], res[-1])


def rob(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)

    if n == 0: return 0

    if n == 1: return nums[0]

    res = [nums[0], max(nums[0], nums[1])]

    for i in range(2, n):
        # ! 这个方程式叫状态转移方程，是DP问题的核心
        res.append(max(res[i - 2] + nums[i], res[i - 1]))


    return res[-1]


# %%
# * Design Circular Queue
# ! And remember when you want to process the elements in order
# ! using a queue might be a good choice
class MyCircularQueue:

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.deque = []
        self.size = k
        self.rear = 0

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if len(self.deque) >= self.size:
            return False
        else:
            self.deque.append(value)
            self.rear += 1
            return True

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if len(self.deque) > 0:
            self.deque.pop(0)
            self.rear -= 1
            return True
        else:
            return False

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if len(self.deque) > 0:
            return self.deque[0]
        else:
            return -1

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if len(self.deque) > 0:
            return self.deque[self.rear - 1]
        else:
            return -1

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return 0 == self.rear

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return self.rear == self.size


# %%
# * Perfect Squares
# ! 四平方和定理: 任意一个正整数均可表示为4个整数的平方和，其实是可以表示为4个以内的平方数之和
def numSquares(self, n):
    # ! 这道题的动态规划(DP)的解法，牛逼
    """
    :type n: int
    :rtype: int
    """
    if n==0:
        return 0
    output = [0x7fffffff]*(n+1)
    output[0] = 0
    output[1] = 1
    for i in range(2,n+1):
        j = 1
        while(j*j<=i):
            # ! 这个状态转移方程真的牛逼
            output[i] = min(output[i],output[i-j*j]+1)
            j = j+1
    return output[n]


# %%
# * Evaluate Reverse Polish Notation
def evalRPN(self, tokens):
    # ! 求解逆波兰式，主要利用栈
    """
    :type tokens: List[str]
    :rtype: int
    """
    stack = []

    for item in tokens:
        # print(stack)
        if item.isdigit():
            stack.append(int(item))

        if item[0] == '-' and len(item) > 1 and item[1:].isdigit():
            stack.append(int(item))

        if item == '*':
            num1 = stack.pop()
            num2 = stack.pop()

            stack.append(num1 * num2)

        if item == '/':
            num1 = stack.pop()
            num2 = stack.pop()

            stack.append(int(num2 / num1))


        if item == '+':
            num1 = stack.pop()
            num2 = stack.pop()

            stack.append(num1 + num2)

        if item == '-':
            num1 = stack.pop()
            num2 = stack.pop()

            stack.append(num2 - num1)



    return stack[0]


# %%
# * String Without AAA or BBB
# ! 使用贪心算法(Greedy)进行解题
def strWithout3a3b(self, A, B):
    # ! 迭代法
    """
    :type A: int
    :type B: int
    :rtype: str
    """
    res = ''
    if A == 0:
        return 'b' * B

    if B == 0:
        return 'a' * A

    while A > 0 and B > 0:
        print(A)
        print(B)
        if A > B:
            res += 'aab'
            A -= 2
            B -= 1

        elif B > A:
            res += 'bba'
            B -= 2
            A -= 1
        else:
            res += 'ab'
            B -= 1
            A -= 1


    if A: return res + ('a' * A)

    if B: return res + ('b' * B)

    return res


def strWithout3a3b(self, A, B):
    # ! 递归法
    """
    :type A: int
    :type B: int
    :rtype: str
    """
    if A == 0:
        return 'b' * B
    elif B == 0:
        return 'a' * A
    elif A == B:
        return 'ab' + self.strWithout3a3b(A-1, B-1)
    elif A > B:
        return 'aab' + self.strWithout3a3b(A-2, B-1)
    else:
        return 'bba' + self.strWithout3a3b(A-1, B-2)


# %%
# * Binary Tree Preorder Traversal
# ! 前序遍历
'''
When you meet a tree problem,
ask yourself two questions:
can you determine some parameters to help the node know the answer of itself ?
Can you use these parameters and the value of the node itself to determine what should be the parameters parsing to its children ?
If the answers are both yes, try to solve this problem using a "top-down" recursion solution
'''

def preorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    self.res = []

    def pre_traverse(node):
        if node:
            self.res.append(node.val)
            pre_traverse(node.left)
            pre_traverse(node.right)


    pre_traverse(root)

    return self.res


# %%
# * Populating Next Right Pointers in Each Node
# ! 广度优先搜索
class Solution:
# @param root, a tree link node
# @return nothing
def connect(self, root):
    if not root: return
    root.next = None
    bfs = []
    lastnumber = 2
    count = 0
    if root.left and root.right:
        bfs.append(root.left)
        bfs.append(root.right)
    else:
        return

    while bfs:
        left_node = bfs.pop(0)
        right_node = bfs.pop(0)
        lastnumber -= 2

        if left_node.left:
            count += 1
            bfs.append(left_node.left)
        if left_node.right:
            count += 1
            bfs.append(left_node.right)
        if right_node.left:
            count += 1
            bfs.append(right_node.left)
        if right_node.right:
            count += 1
            bfs.append(right_node.right)

        if lastnumber == 0:
            left_node.next = right_node
            right_node.next = None
            lastnumber = count
            count = 0
        else:
            left_node.next = right_node
            right_node.next = bfs[0]


# %%
# * Populating Next Right Pointers in Each Node II
# ! 广度优先搜索
class Solution:
# @param root, a tree link node
# @return nothing
def connect(self, root):
    if not root: return
    root.next = None
    bfs = [root]
    lastnumber = 1
    count = 0

    while bfs:
        node = bfs.pop(0)
        lastnumber -= 1

        if node.left:
            count += 1
            bfs.append(node.left)

        if node.right:
            count += 1
            bfs.append(node.right)

        if lastnumber == 0:
            node.next = None
            lastnumber = count
            count = 0

        else:
            node.next = bfs[0]


# %%
# *
