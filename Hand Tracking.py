import cv2
import time
import mediapipe as mp #google tarafından oluşturulan bir kütüphane

cap=cv2.VideoCapture(0) #kamera aç

mpHand=mp.solutions.hands #el takibine odaklı modülü çağırdık ve mpHand fonksiyonu olarak tanımladık

hands=mpHand.Hands() # .Hands modülünde çeşitli fonksşyonlar atayabiliyoruz. örnek: max_num_hands=1 burda max el sayısını girmiş olduk

mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, img=cap.read() #success=görüntüyü alıp alamadığımızı gösteriyor (Varible Explorer kısmında true yada false yazar), img de kameradan read ettiğimiz görüntü
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB yi BGR ye çevirdik

    results=hands.process(imgRGB) #hand processes el takip işlemini baştalır. Elleri tespit etmek için kullanılır. results bizim atadığımız bir değişkendir
    print(results.multi_hand_landmarks) #Elimizde bulunan eklemlerin koordinatları çıkmaya başladı.

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark): #enumerate: xyz leri döndürüp lm içine atıyor ve bu xyz koordinatlarının hangi eklem olduğunu id değişkeninin içine atıyor.
                print(id, lm) # bize 0-1-2-3....-20 ye kadar eklem bölgelerin koordinatlarını bastıracak
                h, w, c = img.shape #Yükseklik, genişlik ve color olarak atadık. Biz bunları koordinat eksenlerine çevireceğiz (cx,cy...)
                cx, cy=int(lm.x*w), int(lm.y*h)
                #bilek (id si 0 numara)
                if id==4: #(id ye 4 koymamızın nedeni baş parmağımızın ucunu temsil etmesi. Eğer 20 yazarsak serçe parmağı almış oluruz.
                    #resmi ekledik, koordinat belirledik, boyut, renk, içi dolu olması için FILLED dedik
                    cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)
    #fps (algoritmanın ne kadar hızlı çalışıp çalışmadığını görmek için ekliyoruz)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, "FPS: "+str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 5)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
    cv2.imshow("img", img)
    cv2.waitKey(1)
