import driverCam
import PySimpleGUI as sg
from PIL import Image, ImageTk
import io
import random
import os
import cv2

class Card():
    def __init__(self, rankvalue, rankname, suit, img):
        self.rankvalue = rankvalue
        self.rankname = rankname
        self.suit = suit
        self.img = img

    def __lt__(self, other):
        return self.rankvalue < other.rankvalue

    def __gt__(self, other):
        return self.rankvalue > other.rankvalue

    def __eq__(self, other):
        return self.rankvalue == other.rankvalue

    def __str__(self):
        return self.rankname + " of " + self.suit


def create_deck():
    # creates a deck of cards, which are instances of the Card class
    # cards have their own </>/==, with the lt/gt/eq methods
    # cards have their own printable version, with the str method
    deck = []
    clubs = []
    spades = []
    hearts = []
    diamonds = []
    n = 13

    clubs_directory = 'deck_img/clubs/'
    spades_directory = 'deck_img/spades/'
    hearts_directory = 'deck_img/hearts/'
    diamonds_directory = 'deck_img/diamonds/'

    suits = ("Clubs", "Spades", "Hearts", "Diamonds")
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace")
    for s in suits:
        if s == "Clubs":

            for path in os.listdir(clubs_directory):
                # check if current path is a file
                if os.path.isfile(os.path.join(clubs_directory, path)):
                    clubs.append(os.path.join(clubs_directory, path))

            for i, r in enumerate(ranks):
                deck.append(Card(i + 2, r, s, clubs[(i + 5) % n]))

        if s == "Spades":

            for path in os.listdir(spades_directory):
                # check if current path is a file
                if os.path.isfile(os.path.join(spades_directory, path)):
                    spades.append(os.path.join(spades_directory, path))

            for i, r in enumerate(ranks):
                deck.append(Card(i + 2, r, s, spades[(i + 5) % n]))

        if s == "Hearts":

            for path in os.listdir(hearts_directory):
                # check if current path is a file
                if os.path.isfile(os.path.join(hearts_directory, path)):
                    hearts.append(os.path.join(hearts_directory, path))

            for i, r in enumerate(ranks):
                deck.append(Card(i + 2, r, s, hearts[(i + 5) % n]))

        if s == "Diamonds":

            for path in os.listdir(diamonds_directory):
                # check if current path is a file
                if os.path.isfile(os.path.join(diamonds_directory, path)):
                    diamonds.append(os.path.join(diamonds_directory, path))
            for i, r in enumerate(ranks):
                deck.append(Card(i + 2, r, s, diamonds[(i + 5) % n]))

    return deck


player_score = 0
computer_score = 0


def find_winner(card1, card2):
    if card1 > card2:
        return "Player"
    if card1 < card2:
        return "Computer"
    else:
        return "No one"


def shuffle_cards(deck):
    return random.shuffle(deck)


def get_p1_card(deck):
    p1_card = deck.pop()
    return p1_card


def get_p2_card(deck):
    p2_card = deck.pop()
    return p2_card


# Add input card to function
inputRank = ""
inputSuit = ""


def checkCard(deck, inputRank, inputSuit, text):
    k = 0
    for i in range(len(deck)):
        if deck[i].rankname == inputRank and deck[i].suit == inputSuit:
            card = deck.pop(i)
            # print(f'Match found after {k} iterations')            # Debugging
            # print(f'Card found: {card.rankname} of {card.suit}')  # Debugging
            k = 0
            return card
        if (k+1) == len(deck):
            text.update("You have already played this card. Please scan a new card")
            k=0
            real_deck.pop()
            real_deck.pop()
            break
        else:
           # print(f'No match, iteration: {k+1}, Retrying...') # Debugging
           k += 1


camOn = False


def CameraDriver(cam):
    train_ranks = driverCam.load_ranks("Image_feature/Rank/")
    train_suits = driverCam.load_suits("Image_feature/Suit/")
    f = 0
    while True:
        check, frame = cam.read()
        f += 1
        #cv2.imshow("Scanning...", frame)
        #cv2.waitKey(1)

        if f == 5:
            f = 0
            R_img, S_img = driverCam.card_finder(frame)
            if R_img != -1 and S_img != -1:
                for i in range(len(R_img)):
                    r, s, r_check, s_check = driverCam.match_card(R_img[i], S_img[i], train_ranks, train_suits)
                    if r_check < 1300 and s_check < 1300 or r == "Queen":  # Problems with queens, more complicated contour
                        if r != "Unknown" and s != "Unknown":
                            print(r, s, r_check, s_check)
                            inputRank = r
                            inputSuit = s
                            print(f'  input rank: {inputRank}')
                            print(f' input rank: {inputSuit}')
                            if r and s not in real_deck:
                                real_deck.append(r)
                                real_deck.append(s)
                                #camText.update(f'Scanned card: {inputRank}, {inputSuit}')
                                return inputRank, inputSuit
                            else:
                                break
# Call functions
deck = create_deck()
deck_player = create_deck()
shuffle_cards(deck)

right_layout = [
    [sg.Text("Computer")],
    [sg.Image(size=(150, 250), key='-IMAGE1-')],
    [sg.Text("Player")],
    [sg.Image(size=(150, 250), key='-IMAGE2-')],
    [sg.Text(f"Player Score: {player_score}", key='-PLAYERSCORE-')],
    [sg.Text(f"Computer Score: {computer_score}", key='-COMPUTERSCORE-')],
]

left_layout = [
    [sg.Text("Higher lower game in python GUI")],
    [sg.Button("Scan Your Card")],
    [sg.Button("Restart Game")],
    [sg.Button("Scan Deck")],
    [sg.Button("Switch Cam")],
    [sg.Text("Console of game: ", key='-CONSOLE-')],
    [sg.Multiline("Scan your card to start the game", size=(50, 5), key='-TEXTBOX-')],
    [sg.Image(size=(300, 300), filename='', key='-cam-')],
    #[sg.Text(f"Scanned card: {inputRank}, {inputSuit}", key='-camText-')],
    [sg.Button('Exit', button_color=('white', 'firebrick3'))]

]

layout = [
    [sg.Column(left_layout),
     sg.VSeperator(),
     sg.Column(right_layout),
     ],
]

window = sg.Window("Higher lower", layout, finalize=True)
text = window['-TEXTBOX-']
img = window['-IMAGE1-']
img2 = window['-IMAGE2-']
camWindow = window['-cam-']
#camText = window['-camText-']
text_player_score = window['-PLAYERSCORE-']
text_computer_score = window['-COMPUTERSCORE-']

real_deck = []

def UpdateCam(cam):
    check, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # let img be the PIL image
    img_cam = Image.fromarray(gray)  # create PIL image from frame
    im_resize = img_cam.resize((300, 300))
    bio = io.BytesIO()  # a binary memory resident stream
    im_resize.save(bio, format='PNG')  # save image as png to it
    imgbytes = bio.getvalue()  # this can be used by OpenCV hopefully
    #camWindow.update(data=imgbytes)
    return imgbytes

def mainDriver(deck, deck_player, player_score, computer_score, camnr):
    cam = cv2.VideoCapture(camnr, cv2.CAP_DSHOW)
    while True:
        event, values = window.read(timeout=20)
        c = UpdateCam(cam)
        camWindow.update(data=c)
        if event == "Scan Your Card" or event == sg.WIN_CLOSED:
            #cam.release()
            c = CameraDriver(cam)
            while len(deck) >= 2:
                cpu_card = get_p1_card(deck)
                # Scan and check player card from Camera, Add card rank & suit to array
                player_card = checkCard(deck_player, c[0], c[1], text)
                if player_card is None:
                    break
                # Remove card from newly created array
                real_deck.pop()
                real_deck.pop()
                print(f'Player card: {player_card}')
                winner = find_winner(player_card, cpu_card)

                # Display game state in textbox
                text.update(f'Computer draws a {cpu_card}' + '\n')
                text.update(f'Player draws a {player_card}' + '\n', append=True)
                text.update(f'{winner} wins this round' + '\n', append=True)

                if winner == "Player":
                    player_score += 1
                    text_player_score.update(f'Player Score: {player_score}')

                if winner == "Computer":
                    computer_score += 1
                    text_computer_score.update(f'Computer Score: {computer_score}')

                # Change img to card that is in play
                cpu_img = cpu_card.img
                # Resize img
                size = (150, 250)
                im = Image.open(cpu_img)
                im = im.resize(size, resample=Image.BICUBIC)
                image = ImageTk.PhotoImage(image=im)
                # Update img in window
                img.update(data=image)

                # Change img to card that is in play
                player_img = player_card.img
                # Resize img
                size = (150, 250)
                im2 = Image.open(player_img)
                im2 = im2.resize(size, resample=Image.BICUBIC)
                image2 = ImageTk.PhotoImage(image=im2)
                # Update img in window
                img2.update(data=image2)

                if len(deck) <= 1:
                    if player_score > computer_score:
                        text.update(f'Deck is empty, You win!', append=True)
                    if player_score < computer_score:
                        text.update(f'Deck is empty. The computer wins, better luck next time!', append=True)
                    if player_score == computer_score:
                        text.update(f'Deck is empty. The game ended in a draw, no winners this time!')
                break

        if event == "Restart Game":
            text.update("Game restarted!")
            img.update(data=None)
            img2.update(data=None)
            deck = create_deck()
            deck_player = create_deck()
            shuffle_cards(deck)
            player_score = 0
            computer_score = 0
            text_player_score.update(f'Player Score: {player_score}')
            text_computer_score.update(f'Computer Score: {computer_score}')

        if event == "Scan Deck":
            driverCam.card_scanner(cam)

        if event == "Switch Cam":
            cam.release()
            if camnr == 0:
                camnr = 1
                cam = cv2.VideoCapture(camnr, cv2.CAP_DSHOW)

            else:
                camnr = 0
                cam = cv2.VideoCapture(camnr, cv2.CAP_DSHOW)
        if event in ('Exit', None):
            break




mainDriver(deck, deck_player, player_score, computer_score,0)
