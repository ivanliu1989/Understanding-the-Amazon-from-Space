# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 06:10:16 2017

@author: Ivan Liu
"""

##--- helpe functions  -------------


def change_images(images, agument):

    num = len(images)
    if agument == 'left-right' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,1)

    if agument == 'up-down' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,0)

    # if agument == 'rotate':
    #     for n in range(num):
    #         image = images[n]
    #         images[n] = randomRotate90(image)  ##randomRotate90  ##randomRotate
    #

    if agument == 'transpose' :
        for n in range(num):
            image = images[n]
            images[n] = image.transpose(1,0,2)


    if agument == 'rotate90' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,1)


    if agument == 'rotate180' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,-1)


    if agument == 'rotate270' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,0)

    return images


#------------------------------------------------------------------------------------------------------------
#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
def find_f_measure_threshold1(probs, labels, thresholds=None):

    #f0 = fbeta_score(labels, probs, beta=2, average='samples')  #micro  #samples
    def _f_measure(probs, labels, threshold=0.5, beta=2 ):

        SMALL = 1e-12 #0  #1e-12
        batch_size, num_classes = labels.shape[0:2]

        l = labels
        p = probs>threshold

        num_pos     = p.sum(axis=1) + SMALL
        num_pos_hat = l.sum(axis=1)
        tp          = (l*p).sum(axis=1)
        precise     = tp/num_pos
        recall      = tp/num_pos_hat

        fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
        f  = fs.sum()/batch_size
        return f


    best_threshold =  0
    best_score     = -1

    if thresholds is None:
        thresholds = np.arange(0,1,0.005)
        ##thresholds = np.unique(probs)

    N=len(thresholds)
    scores = np.zeros(N,np.float32)
    for n in range(N):
        t = thresholds[n]
        #score = f_measure(probs, labels, threshold=t)
        score = fbeta_score(labels, probs>t, beta=2, average='samples')  #micro  #samples
        scores[n] = score

    return thresholds, scores



def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.235):

    batch_size, num_classes = labels.shape[0:2]

    best_thresholds = [seed]*num_classes
    best_scores     = [0]*num_classes
    for t in range(num_classes):

        thresholds = [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if  f2 > best_scores[t]:
                best_scores[t]     = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f'%(t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores
#------------------------------------------------------------------------------------------------------------


#precision_recall_curve
def binary_precision_recall_curve(labels, predictions, beta=2):

    precise, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions)
    f2 = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + 1e-12)  #beta=2  #f2 score
    idx = np.argmax(f2)

    return precise, recall, f2, threshold, idx




# write csv
def write_submission_csv(csv_file, predictions, thresholds):

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    with open(KAGGLE_DATA_DIR+'/split/test-61191') as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    num_test = len(names)


    assert((num_test,num_classes) == predictions.shape)
    with open(csv_file,'w') as f:
        f.write('image_name,tags\n')
        for n in range(num_test):
            shortname = names[n].split('/')[-1].replace('.<ext>','')

            prediction = predictions[n]
            s = score_to_class_names(prediction, class_names, threshold=thresholds)
            f.write('%s,%s\n'%(shortname,s))







# loss ----------------------------------------
def multi_criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss

#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
#f  = fbeta_score(labels, probs, beta=2, average='samples')
def multi_f_measure( probs, labels, threshold=0.235, beta=2 ):

    SMALL = 1e-6 #0  #1e-12
    batch_size = probs.size()[0]

    #weather
    l = labels
    p = (probs>threshold).float()

    num_pos     = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/(num_pos     + SMALL)
    recall      = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f
