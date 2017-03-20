from sklearn import tree

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Initialize classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier on the model
clf.fit(X, Y)
prediction = clf.predict([172, 73, 42])
print(prediction)

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3
