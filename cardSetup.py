import numpy as np
import cv2
import os
import math
aso = 0
NOICE_DIMENSION = 50

CARD_MAX_AREA = 200000
CARD_MIN_AREA = 25000

RANK_WIDTH = 130
RANK_HEIGHT = 100

suit_WIDTH = 130
suit_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

Rank = []
suit = []

class Train_ranks:
    # Store Database Images of Ranks.

    def __init__(self):
        self.img = []
        self.name = "Placeholder"


class Train_suits:
    # Store Database Images of Suits.

    def __init__(self):
        self.img = []
        self.name = "Placeholder"


def card_finder(img):
    cards_array_rank = []
    cards_array_suit = []

    images_cards = []
    images_zoom = []

    Rank_img = []
    suit_img = []

    contour_finish = []

    # preprocessing image with both pre-made function and finds image contour.
    img_post = image_preprocess(img)
    contour, hierarchy = image_contour(img_post)
    # calls one function to find if it is 4 corners and is a card
    corners_to_cards, card_contour_place = find_corners(contour, hierarchy)
    # if there is no cards return -1
    if corners_to_cards == -1:
        return -1, -1

    # finds the card using corner coordinates, then creates a bounding rectangle around
    # the card and then warps, and cuts it. This gives the left corner image of a card.
    for i in range(len(corners_to_cards)):
        x, y, w, h = bounding_rect(contour[card_contour_place[i]])
        pts = np.float32(corners_to_cards[i])
        image_card = warp_contour(img, pts, w, h)
        images_cards.append(image_card[0:84, 0:37])


    # Then we resize it one more time by dividing it by four, and we get the left
    # top most cropped out image of a card.
    for r in range(len(images_cards)):
        z = cv2.resize(images_cards[r], (0, 0), fx=4, fy=4)
        images_zoom.append(z)

    # We normalize and threshold the final zoomed in image, and
    # We find the contours
    final_pro = []
    for zoom in range(len(images_zoom)):
        normal = cv2.normalize(images_zoom[zoom], images_cards[zoom], 0, 255,cv2.NORM_MINMAX)
        ret, thresh_card = cv2.threshold(normal, 127, 255, 0, cv2.THRESH_BINARY_INV)
        final_pro.append(thresh_card)
        contour_zoom, hierarchy_zoom = cv2.findContours(thresh_card, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # We check the contour area and filter out the ones too big
        # We do this for all the cards
        zoom_contour = []
        for size in range(len(contour_zoom)):
            if cv2.contourArea(contour_zoom[size]) < 10000:
                zoom_contour.append(contour_zoom[size])
        contour_finish.append(zoom_contour)

    # We find which is the contours of the rank and suit
    for card in range(len(contour_finish)):
        rank, suit = find_card_info(contour_finish[card], thresh_card.shape[:2])
        cards_array_rank.append(rank)
        cards_array_suit.append(suit)

    # Checks if the array is empty
    if cards_array_rank == None:
        return -1

    # If the amount of ranks obtained is equal to the amount of suits then,
    # We crop out an image of the ranks and suits.
    if len(cards_array_rank) == len(cards_array_suit):
        for t in range(len(cards_array_rank)):
            if len(cards_array_rank[t]) != 0 and len(cards_array_rank[t]) != 1 and len(cards_array_suit[t]) != 0 and len(cards_array_suit[t]) != 1: #check
                final_rank = crop_contour_rank(cards_array_rank[t], final_pro[t])
                final_suit = crop_contour_suit(cards_array_suit[t], final_pro[t])
                Rank_img.append(final_rank)
                suit_img.append(final_suit)

    else:
        return -1

    return Rank_img, suit_img


def image_preprocess(img):
    # Gray the imaged then thresholding it
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret, thresh = cv2.threshold(img_blur, 127, 255, 0, cv2.THRESH_BINARY)
    return thresh


def image_contour(img):
    contour_sort = []
    hierarchy_sort = []

    # Find the contour of the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If the image have no contours then return empty array
    if len(contours) == 0:
        return [], []

    # We Sort the contours and removes the too small contours
    index_sort = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    for i in index_sort:
        perimeter = cv2.arcLength(contours[i], True)
        if perimeter > NOICE_DIMENSION:
            contour_sort.append(contours[i])
            hierarchy_sort.append(hierarchy[0][i])
        else:
            continue

    # We return the sorted contours and the hierarchy
    return contour_sort, hierarchy_sort


def find_card_info(contours, card_height):
    # Create variables and find the center of the contours given
    contours_centers = contour_center(contours)
    temp_rank = []
    temp_suit = []
    distance = 10000000000
    # Calculate the distance from the left top corner to the nearest contour,
    # and the closest one is saved as the rank
    for con in range(len(contours_centers)):
        current_dist = math.dist((0, 0), (contours_centers[con][0], contours_centers[con][1]))
        if distance > current_dist:
            if contours[con][0][0][0] != 0 and contours[con][0][0][1] != 0:
                distance = current_dist
                temp_rank = contours[con]

    distance = 10000000000
    distance1 = 10000000000

    # If there is no contours that can represent the rank then return [-1]
    if(len(temp_rank) == 0):
        return [-1], [-1]

    # Finds the center of the contour representing the rank
    M = cv2.moments(temp_rank)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        rank_center = [cX, cY]
    else:
        rank_center = [0, 0]

    # Calculate the distance from the left bottom corner to the nearest contour,
    # and the closest one is saved as the Suit
        for con1 in range(len(contours_centers)):
            current_dist1 = math.dist((card_height[0] / 2, card_height[1] / 2),
                                      (contours_centers[con][0], contours_centers[con][1]))
            if distance1 > current_dist1:
                if contours[con][0][0][0] != 0 and contours[con][0][0][1] != 0:
                    distance1 = current_dist1
                    temp_suit = contours[con1]
        # Finds if there is any almost connected contours to the rank and then sets them together
        current_dist = math.dist((rank_center[0], rank_center[1]),
                                 (contours_centers[con1][0], contours_centers[con1][1]))
        if distance > current_dist > 60:
            distance = current_dist
        if current_dist < 60:
            temp_rank = np.vstack([temp_rank, contours[con1]])
            for melt in range(len(contours_centers)):
                current_dist = math.dist((contours_centers[con1][0], contours_centers[con1][1]),
                                         (contours_centers[melt][0], contours_centers[melt][1]))
                if current_dist < 60:
                    temp_rank = np.vstack([temp_rank, contours[melt]])

    # Returns the array of ranks and suits
    return temp_rank, temp_suit


def find_corners(contour, hierarchy):
    # Declare variables
    contour_cards = []
    card_count = 0
    contour_card = []
    contour_place = []
    contour_outline = []

    # Removes some small contours, and finds the corners of the contours.
    # If the contours have four corners, have no parent contours,
    # and is smaller than the maximum card size and bigger than the minimum card size
    # also counts the card
    for i in range(len(contour)):
        x, y = "", ""

        # Removes some small contours
        perimeter = cv2.arcLength(contour[i], True)
        if perimeter > NOICE_DIMENSION:
            corners = cv2.approxPolyDP(contour[i], 0.04 * perimeter, True)
        else:
            continue

        # checks if four corners and is within the card sizes, and saves the corner coordinates
        if len(corners) == 4 and hierarchy[i][3] == -1 and \
        cv2.contourArea(contour[i]) > CARD_MIN_AREA and \
        cv2.contourArea(contour[i]) < CARD_MAX_AREA:
            contour_place.append(i)
            contour_outline.append(contour[i])
            for c in range(len(corners)):
                index = []
                for s in range(len(corners[c-1])):
                    x = str(corners[c-1][s-1][0])
                    y = str(corners[c-1][s-1][1])

                index.append(x), index.append(y)
                contour_card.append([index])
            contour_cards.append(contour_card)
            contour_card = []
            card_count += 1

    # Returns -1 if there is no cards
    if card_count == 0:
        return -1, -1

    # Returns the contour of the cards and its place in the first contour list
    return contour_cards, contour_place


def bounding_rect(contour):
    # creates a bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def warp_contour(image, pts, w, h):
    temp_rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8 * h and w < 1.2 * h:
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 200
    maxHeight = 300

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp


def contour_center(contour_list):
    contours_center = []
    cX = 0
    cY = 0

    # Finds the center of a list with contours, by using the average with cv2's moments
    for con in range(len(contour_list)):
        M = cv2.moments(contour_list[con])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        contours_center.append([cX, cY])

    # returns the center oqqf all contours in a list
    return contours_center


def crop_contour_rank(contour, img):
    # Crops the rank contour from the image as a new image
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    rank_roi = img[y1:y1 + h1, x1:x1 + w1]
    dim = (RANK_WIDTH, RANK_HEIGHT)
    rank_sized = cv2.resize(rank_roi, dim, 0, 0)

    # Returns a cropped out image of the rank
    return rank_sized


def crop_contour_suit(contour, img):
    # Crops the suit contour from the image as a new image
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    suit_roi = img[y1:y1 + h1, x1:x1 + w1]
    dim = (suit_WIDTH, suit_HEIGHT)
    suit_sized = cv2.resize(suit_roi, dim, 0, 0)

    # Returns a cropped out image of the suit
    return suit_sized


def load_ranks(filepath):
    train_ranks = []
    i = 0

    # loads the images in the Rank directory to a class object
    for Rank in ['Ace', '2', '3', '4', '5', '6', '7',
                 '8', '9', '10', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.png'
        train_ranks[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    # returns the class object train_ranks
    return train_ranks


def load_suits(filepath):
    train_suits = []
    i = 0

    # loads the images in the Suit directory to a class object
    for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.png'
        train_suits[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    # returns the class object train_suits
    return train_suits


def match_card(rank_img, suit_img,  train_ranks, train_suits):

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    # If no cards where found in preprocess functions,
    # if the rank or suit array is zero then there is no cards, so it skips the differencing process
    # and the card will be left as Unknown
    if (len(rank_img) != 0) and (len(suit_img) != 0):

        # Difference the ranks images from each of the train rank images,
        # and store the result with the least difference
        for Trank in train_ranks:
            diff_img = cv2.absdiff(rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img) / 255)

            if rank_diff < best_rank_match_diff:
                best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

        # Difference the suits images from each of the train suit images,
        # and store the result with the least difference
        for Tsuit in train_suits:

            diff_img = cv2.absdiff(suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img) / 255)

            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_name = Tsuit.name

    # Combine best rank match and best suit match to get the card's identity.
    # If the best matches have too high of a difference value, then the card identity
    # is set as Unknown
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    # Return the rank, suit, and the rank and suit score.
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff


def card_detection(cam_nr):
    cam = cv2.VideoCapture(cam_nr, cv2.CAP_DSHOW)
    check, frame = cam.read()

    train_ranks = load_ranks("Image_feature/Rank/")
    train_suits = load_suits("Image_feature/Suit/")

    while True:
        check, frame = cam.read()
        cv2.imshow('video', frame)
        R_img, S_img = card_finder(frame)
        if R_img != -1 and S_img != -1:
            for i in range(len(R_img)):
                r, s, r_check, s_check = match_card(R_img[i], S_img[i], train_ranks, train_suits)
                if r_check < 1300 and s_check < 1300 or r == "Queen": # Problems with queens, more complicated contour
                    if r != "Unknown" and s != "Unknown":
                        print(r, s, r_check, s_check)
                        return r, s

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def card_scanner(cam):
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8',
             '9', '10', 'Jack', 'Queen', 'King']
    suits = ['Spades', 'Diamonds',
             'Clubs', 'Hearts']
    i = 0
    n = -1
    while True:
        check, frame = cam.read()
        R_img, S_img = card_finder(frame)
        if i < 13:
            cv2.putText(frame, ranks[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(frame, "a=next, d=last, q= quit", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(frame, "x=Take IMG, w=Save, s=Redo", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 3, cv2.LINE_AA)
        if i > 12 and n > -1:
            cv2.putText(frame, suits[n], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(frame, "a=next, d=last, q= quit", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(frame, "x=Take IMG, w=Save, s=Redo", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3,cv2.LINE_AA)
        cv2.imshow('video', frame)
        key = cv2.waitKey(1)

        if R_img != -1 and S_img != -1 and R_img != []:
            if key == ord('x'):
                if n == -1:
                    cv2.imshow('video', R_img[0])
                    key1 = cv2.waitKey(0)
                    if key1 == ord('w'):
                        path = os.path.dirname(os.path.abspath(__file__)) + '/Image_feature/Rank/'
                        filename = ranks[i] + '.png'
                        cv2.imwrite(path + filename, R_img[0])
                    if key1 == ord('s'):
                        break

                if n > -1:
                    cv2.imshow('video', S_img[0])
                    key2 = cv2.waitKey(0)
                    if key2 == ord('w'):
                        path = os.path.dirname(os.path.abspath(__file__)) + '/Image_feature/Suit/'
                        filename = suits[n] + '.png'
                        cv2.imwrite(path + filename, S_img[0])
                    if key2 == ord('s'):
                        break

        if key == ord('a'):
            if i != 16:
                i += 1
                if i > 12:
                    n += 1
        if key == ord('d'):
            if i > 0:
                i -= 1
                if n > -1:
                    n -= 1
        if key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return 1
