# import the necessary packages
import numpy as np

def NMS(boxes,classes, probs, overlapThresh,imH,imW):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# For the bunch of division, the type of elements in bbox should be converterd to float 
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	
	
	x1 = boxes[:, 1]*imW
	y1 = boxes[:, 0]*imH
	x2 = boxes[:, 3]*imW
	y2 = boxes[:, 2]*imH
	
	
	
	
	
		

	
	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1+1) * (y2 - y1+1)#By adding 1, we can prevent multiplication on floating point numbers. 
	
	idxs = y2.copy()

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs
	
	#이미지 내 바운딩박스마다 가지고 있는 confidence score을 이용해서 
	#confidnece score를 기준으로 오름차순으로 정렬한다. 
	idxs = np.argsort(idxs)#index로 저장되어 있다. 
	
	print(idxs)
	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]#i는 특정 (x1,y1,x2,y2)박스의 confidence값을 가리키는 index이다.
		pick.append(i)# confidence score 중 가장 큰 값을 리스트에 추가한다. 

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		#idxs[:last]는 마지막 값을 제외한 나머지 값들 즉, <last
		#np.maximum은 두 배열을 비교하기 위해 둘중 작은 배열을 element-wise한 다음에
		#각각 배열을 순서대로 원소를 비교한 다음 큰 값을 새 배열에 넣는다. 
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
		

	# return only the bounding boxes that were picked
	return boxes[pick],probs[pick],classes[pick]
