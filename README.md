# 📄 InsightCV — AI-Powered Resume Analyzer

InsightCV هو تطبيق ويب مبني بـ Python و Streamlit يحلل السير الذاتية (PDF) باستخدام تقنيات معالجة اللغة الطبيعية (NLP)، ويقدّم للمستخدم توصيات مهارات، دورات تدريبية، ومطابقة دلالية (Semantic Matching) مع وصف وظيفي معيّن.

## ✨ المميزات

- **تحليل السيرة الذاتية**: استخراج تلقائي للاسم، الإيميل، رقم الجوال، المهارات، وعدد الصفحات.
- **تصنيف المجال الوظيفي**: يحدد المجال المتوقع للمستخدم (Data Science, Web Dev, Android, iOS, UI/UX, Cybersecurity, DevOps, Data Analysis, Digital Marketing, Game Dev) بناءً على المهارات المستخرجة.
- **توصيات مهارات ودورات**: يقترح مهارات إضافية ودورات تعليمية مناسبة لكل مجال.
- **درجة قوة السيرة الذاتية (Resume Score)**: تقييم من 0 إلى 100 بناءً على اكتمال الأقسام، عدد المهارات، وعدد المشاريع.
- **المطابقة الدلالية (Semantic Job Matching)**: باستخدام TF-IDF + Cosine Similarity لمقارنة السيرة الذاتية مع وصف وظيفي حقيقي.
- **تحليل جماعي للجامعات (University Batch Analysis)**: رفع عدة سير ذاتية دفعة واحدة، مع رسوم بيانية لجاهزية الطلاب لسوق العمل.
- **لوحة تحكم إدارية (Admin Dashboard)**: عرض كل بيانات المستخدمين والتقييمات مع رسوم بيانية تفاعلية.
- **نظام تقييم (Feedback)**: يسمح للمستخدمين بتقييم التطبيق وترك تعليقات.
- **واجهة Dark Mode** بتصميم احترافي متجانس.

## 🛠️ التقنيات المستخدمة

- **Python** & **Streamlit** — واجهة التطبيق
- **pyresparser** & **pdfminer3** — استخراج بيانات السيرة الذاتية من PDF
- **scikit-learn** (TF-IDF, Cosine Similarity) — المطابقة الدلالية
- **MySQL** (عبر `pymysql`) — تخزين بيانات المستخدمين والتقييمات
- **Plotly** — الرسوم البيانية التفاعلية
- **NLTK** — معالجة اللغة الطبيعية

## 🚀 كيفية التشغيل محليًا

```bash
# 1. إنشاء بيئة افتراضية
python -m venv venvapp
venvapp\Scripts\activate      # ويندوز
source venvapp/bin/activate   # ماك/لينكس

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. تشغيل التطبيق
python -m streamlit run App.py
```

## 📸 لقطات من التطبيق

<!-- ضع صور السكرينشوت هنا (شوف تعليمات الإضافة بالأسفل) -->

![الصفحة الرئيسية](screenshots/home.png)
![تحليل السيرة الذاتية](screenshots/analyze.png)
![لوحة الأدمن](screenshots/admin.png)

## 🔐 الوصول للأقسام المحمية

| القسم | بيانات الدخول |
|---|---|
| University | Access Code: `university2026` |
| Admin | Username: `admin` / Password: `admin@resume-analyzer` |

> ⚠️ يُنصح بتغيير هذه البيانات قبل أي نشر فعلي (production).

## 📁 هيكل المشروع

```
App/
├── App.py                 # الملف الرئيسي للتطبيق
├── Courses.py              # قوائم الدورات والفيديوهات التعليمية
├── requirements.txt         # المكتبات المطلوبة
├── Logo/                    # شعارات وأيقونات
├── Uploaded_Resumes/        # مجلد حفظ السير الذاتية المرفوعة
└── screenshots/              # صور توثيقية للـ README
```

---

<p align="center">Made with ❤️ using Streamlit</p>
# InsightCV
