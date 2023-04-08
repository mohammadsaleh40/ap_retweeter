# ربات توییتری (و تلگرامی) آموزش و پرورش


## موارد مورد نیاز برای اجرای این برنامه

۱. دریافت api [از این طریق](https://developer.twitter.com/).
\
۲. یک سرور برای اجرا که سایت [pythonanywhre](https://www.pythonanywhere.com/) گزینه رایگان خوبیه.
\
۳.یک عدد ربات تلگرامی

## کتابخانه‌های مهم مورد نیاز
1.tensorflwo
\
2.tweepy
\
3.hazm 

کتابخونه hazm برای پردازش اولیه و تمیز سازی متن هست این رو اگر بخواین تو [pythonanywhre](https://www.pythonanywhere.com/) باهاش کار کنین حتما باید نصب کنین.


# اجرا

## چالش‌های اجرا
برای این که هوش مصنوعی روی سرورهای سایت [pythonanywhre](https://www.pythonanywhere.com/) کار کنه نیاز بود که حتما کتابخونه tensorflow کار کنه ولی این کتابخونه وقتی نصب می‌شه حجمش از 1GB هم بیشتره که نمی‌شه روی حالت رایگان سرورهای این سایت نصب بشه، ولی معلوم شد که وقتی پایتون ورژن 3.8 رو ازش فراخوانی می‌کنیم.
برای سبک تر شدن اجرای برنامه بعضی از قسمت‌ها مثل خود مدل هوش مصنوعی و توکنایز کننده رو از قبل آماده کردیم و فراخوانی می‌کنیم که فایل‌هاش تو پوشه pickle_x هست.

## روش اجرا
۱. نیاز داریم توکن‌های تلگرام و توییتر رو ثبت کنیم.
\
۲. کتابخونه tweepy مثلا روی [pythonanywhre](https://www.pythonanywhere.com/) نصب هست ولی مشکل داره باید دوباره (روی پایتون 3.8) نصبش کنین. از این کد باید استفاده کنین:
`pip3.8 install git+https://github.com/tweepy/tweepy.git`
\
۳.فایل‌ها و پوشه‌های این ریپو رو توش آپلود می‌کنین.
\
۴.تابع retweet_ap رو از فایل retweet_robot اجرا می‌کنین.

هر وقت این تابع اجرا بشه توییت‌هایی که در پریروز زده شده استخراج و قضاوت می‌شه بعدا ۱۰ تای برترش ریتوییت می‌شه.
## ضرورت 
چک کردن مداوم فضای مجازی وقت زیادی رو از ما می‌گیره و گرایش‌هایی که با نظرات شخصی مون به وجود می‌یاد ممکنه دید ما رو نسبت به اطراف اریب کنه. اگر به کمک یک ماشین خودکار که شرط‌های منطقی برای خودش داره بخوایم به اطراف نگاه کنیم هم می‌تونیم در وقت مون صرفه جویی کنیم و هم از اریب شدن اطلاعاتی که جمع آوری می‌کنیم جلوگیری می‌کنه.
## هدف
متأسفانه آموزش و پرورش در کشور ما رسانه قوی نداره و هدف مند نیست به همین خاطر نیاز هست هم اطلاعات خوبی در این باره استخراج بشه و هم تلاش بشه روی خبرها و اتفاقات و مباحث مهم آموزش و پرورش تأکید بیشتری بشه. در این پروژه تلاش می‌کنیم قدم کوچکی در این راه برداریم.

