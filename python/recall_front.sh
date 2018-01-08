#!/bin/bash
echo "recall_rate right starts here:">>recall_result.txt
boundary=0.1
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.2
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.3
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.4
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.5
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.6
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.7
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.8
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
boundary=0.9
echo "current boundary is: "$boundary >>recall_result.txt
echo "positive folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/positive $boundary --center_only --gpu
echo "negative folder: ">>recall_result.txt
python recall_right.py /home/capstone/adaboost/adaboost_test/right_test/negative $boundary --center_only --gpu
