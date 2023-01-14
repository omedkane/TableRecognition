import cv2
import pytesseract as tess


def ocrTable(imagePath: str, contourMaskSize = 4):
    # ? Au cas où tesseract ne détecte pas le chemin d'accès au données tessdata, le spécifier ici !
    # tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'
    # ? la configuration de tesseract, soient le mode d'océrisation 3 et le mode de segmentation de page 6
    tess_config = '--oem 3 --psm 6'

    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 127, 255, 0)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def getCoordinates(contour: tuple):
        min_X = contour[0][0][0]
        max_X = 0
        min_Y = contour[0][0][1]
        max_Y = 0
        for sub in contour:
            x = sub[0][0]
            y = sub[0][1]
            if y < min_Y:
                min_Y = y
            if y > max_Y:
                max_Y = y
            if x < min_X:
                min_X = x
            if x > max_X:
                max_X = x

        return [min_X, max_X, min_Y, max_Y]

    def getDimensions(contour: tuple):
        min_X, max_X, min_Y, max_Y = getCoordinates(contour)
        return [max_X - min_X, max_Y - min_Y]

    def areEqualDimensions(contour1: tuple, contour2: tuple, safeMargin=0):
        assert safeMargin >= 0
        width1, height1 = getDimensions(contour1)
        width2, height2 = getDimensions(contour2)

        return (abs(width1 - width2) <= safeMargin) and (abs(height1 - height2) <= safeMargin)

    lastCell = contours[2]  # ? dernière cellule de la dernière ligne
    table = contours[1]
    filteredContours = [
        contour for contour in contours if areEqualDimensions(lastCell, contour)]

    _, _, fcY1, fcY2 = getCoordinates(lastCell)
    lastRowCells = [lastCell]

    for contour in filteredContours[1:]:
        _, _, y1, y2 = getCoordinates(contour)
        if y1 == fcY1 and y2 == fcY2:
            lastRowCells.append(contour)
        else:
            break

    img = cv2.drawContours(img, filteredContours, -1, (255, 255, 255), contourMaskSize)

    # ? Pour une cellule donnée cette fonction océrise toute la region verticale correspondante sauf le nom des vars
    def getVerticalRegionText(contour):
        zy2 = getCoordinates(filteredContours[-1])[-1]
        x1, x2, _, y2 = getCoordinates(contour)
        crop = img[zy2:y2, x1:x2]
        text: str = tess.image_to_string(crop, config=tess_config)
        # ? zeros (0s) are sometimes identified as °
        return text.replace('°', '0').splitlines()

    # ? Cherche les noms des variables (première ligne)
    _, tableX2, _, _ = getCoordinates(table)
    zx1, _, zy1, zy2 = getCoordinates(filteredContours[-1])

    crop = img[zy1:zy2, zx1:tableX2]

    text: str = tess.image_to_string(crop, config=tess_config)
    variables = text[:-1].split(' ') # ? les mots sont séparés par des espaces, d'où split() 

    labels = getVerticalRegionText(lastRowCells[-1])
    observations = []

    for contour in lastRowCells[:-1]:
        verticalRegion = getVerticalRegionText(contour)
        verticalRegion = [int(x) for x in verticalRegion]
        observations.append(verticalRegion)

    # ? renverse la matrice pour avoir le bon ordre.
    observations.reverse()
    
    refinedObservations = []
    for i in range(len(labels)):
        refinedObservations.append([obs[i] for obs in observations])

    # print(labels)

    # cv2.imshow("contours", crop)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return (variables, labels, refinedObservations)
