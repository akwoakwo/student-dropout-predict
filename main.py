import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

st.title("Aplikasi Prediksi Mahasiswa Berstatus Dropout")

selected = option_menu (
    menu_title=None,
    options=["Input Data","Preprocessing","Klasifikasi","Prediksi"],
    icons=["database","gear","hourglass","search"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Input Data":
    st.title(f"📊{selected}")

    data = pd.read_csv("dataset.csv")
    data.rename(columns = {"Nacionality": "Nationality",
                           "Mother's qualification": "Mother_qualification",
                           "Father's qualification": "Father_qualification",
                           "Mother's occupation": "Mother_occupation",
                           "Father's occupation": "Father_occupation",
                           "Age at enrollment": "Age"}, inplace = True
                )
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('(', '')
    data.columns = data.columns.str.replace(')', '')
    
    col = ['Marital_status', 'Application_mode', 'Application_order', 'Course',
            'Daytime/evening_attendance', 'Previous_qualification', 'Nationality',
            'Mother_qualification', 'Father_qualification', 'Mother_occupation',
            'Father_occupation', 'Displaced', 'Educational_special_needs', 'Debtor',
            'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
            'International', 'Target'
        ]

    data[col] = data[col].astype('category')
    
    st.write(data)
    
    st.session_state.data = data

if selected == "Preprocessing":
    st.title(f"⚙️{selected}")
    
    if "data" in st.session_state and st.session_state.data is not None:
        data = st.session_state.data.copy()
        st.write("Data:")
        st.dataframe(data)
        
        if st.button("Transformasi Tipe Data Kelas"):
            encoder = OrdinalEncoder()
            data['Target_encoded'] = OrdinalEncoder(categories = [['Dropout', 'Enrolled', 'Graduate']]).fit_transform(data[['Target']])
            data.drop('Target', axis = 1, inplace = True)
            
            joblib.dump(encoder, 'model/encoder.pkl')
            
            st.write("Data Setelah Dilakukan Transformasi")
            st.dataframe(data)
        
            st.session_state.data = data
        
        if st.button("Korelasi Fitur Tipe Data Kategorikal"):
            data = st.session_state.data.copy()
            
            cats = ['Marital_status', 'Application_mode', 'Application_order',
                    'Course','Daytime/evening_attendance', 'Previous_qualification',
                    'Nationality','Mother_qualification', 'Father_qualification',
                    'Mother_occupation', 'Father_occupation', 'Displaced',
                    'Educational_special_needs', 'Debtor','Tuition_fees_up_to_date',
                    'Gender', 'Scholarship_holder','International'
                    ]
            
            p_value = []
            for col in cats:
                crosstable = pd.crosstab(index = data[col],
                                        columns = data['Target_encoded'])
                p = chi2_contingency(crosstable)[1]
                p_value.append(p)

            chi2_result = pd.DataFrame({
                'Variable': cats,
                'P_value': [round(ele, 5) for ele in p_value]
            })

            chi2_result = chi2_result.sort_values('P_value')
            
            st.write("Sebagian besar nilai-p mendekati nol, kecuali tiga variabel 'Nationality', 'International', 'Educational_special_needs' dengan nilai-p yang sangat tinggi (0,24, 0,53, 0,73), yang menunjukkan tidak adanya hubungan yang signifikan secara statistik antara ketiga fitur ini dan label. Saya akan mengecualikan ketiganya dari pemodelan.")
            st.write(chi2_result)
            
            data = data.drop(['Nationality', 'International', 'Educational_special_needs'], axis = 1)
            
            st.write(data)
            st.session_state.data = data
        
        if st.button("Korelasi Fitur Tipe Data Numerik"):
            data = st.session_state.data.copy()
            
            st.write("Data Nilai Mahasiswa Dari 2 Semester Akan Dijadikan Rata - Rata Menggunakan Kode")
            code = '''
                stud_selected['avg_credited'] = stud_selected[['Curricular_units_1st_sem_credited',
                                'Curricular_units_2nd_sem_credited']].mean(axis = 1)
                stud_selected['avg_enrolled'] = stud_selected[['Curricular_units_1st_sem_enrolled',
                                'Curricular_units_2nd_sem_enrolled']].mean(axis = 1)
                stud_selected['avg_evaluations'] = stud_selected[['Curricular_units_1st_sem_evaluations',
                                'Curricular_units_2nd_sem_evaluations']].mean(axis = 1)
                stud_selected['avg_approved'] = stud_selected[['Curricular_units_1st_sem_approved',
                                'Curricular_units_2nd_sem_approved']].mean(axis = 1)
                stud_selected['avg_grade'] = stud_selected[['Curricular_units_1st_sem_grade',
                                'Curricular_units_2nd_sem_grade']].mean(axis = 1)
                stud_selected['avg_without_evaluations'] = stud_selected[['Curricular_units_1st_sem_without_evaluations',
                                'Curricular_units_2nd_sem_without_evaluations']].mean(axis = 1)
            '''
            st.code(code, language="python")
            
            data['avg_credited'] = data[['Curricular_units_1st_sem_credited',
                            'Curricular_units_2nd_sem_credited']].mean(axis = 1)
            data['avg_enrolled'] = data[['Curricular_units_1st_sem_enrolled',
                            'Curricular_units_2nd_sem_enrolled']].mean(axis = 1)
            data['avg_evaluations'] = data[['Curricular_units_1st_sem_evaluations',
                            'Curricular_units_2nd_sem_evaluations']].mean(axis = 1)
            data['avg_approved'] = data[['Curricular_units_1st_sem_approved',
                            'Curricular_units_2nd_sem_approved']].mean(axis = 1)
            data['avg_grade'] = data[['Curricular_units_1st_sem_grade',
                            'Curricular_units_2nd_sem_grade']].mean(axis = 1)
            data['avg_without_evaluations'] = data[['Curricular_units_1st_sem_without_evaluations',
                            'Curricular_units_2nd_sem_without_evaluations']].mean(axis = 1)
            
            st.write("Heatmap Korelasi Fitur Numerik")
            st.image("korelasi.png")
        
            st.session_state.data = data
            st.write(data)
        
        if st.button("Mendeteksi Outlier Pada Data"):
            data = st.session_state.data.copy()
            st.write("Terdapat Mahasiswa Dengan Nilai Rata - Rata 0 Namun Memiliki Kelas 'Graduate'")
            
            outlier = data.loc[(data['avg_approved'] == 0) & (data['Target_encoded'] == 2)]
            st.write(outlier)
            
            data = data.drop(outlier.index)
            drop_cols = ['Unemployment_rate', 'Inflation_rate',
                        'avg_credited', 'avg_evaluations',
                        'Curricular_units_1st_sem_credited',
                        'Curricular_units_1st_sem_enrolled',
                        'Curricular_units_1st_sem_evaluations',
                        'Curricular_units_1st_sem_approved',
                        'Curricular_units_1st_sem_grade',
                        'Curricular_units_1st_sem_without_evaluations',
                        'Curricular_units_2nd_sem_credited',
                        'Curricular_units_2nd_sem_enrolled',
                        'Curricular_units_2nd_sem_evaluations',
                        'Curricular_units_2nd_sem_approved',
                        'Curricular_units_2nd_sem_grade',
                        'Curricular_units_2nd_sem_without_evaluations']
            data = data.drop(columns=drop_cols, errors='ignore')
            
            st.session_state.data = data
            st.write(data)
        
        if st.button("Normalisasi Data Dengan MinMaxScaler"):
            data = st.session_state.data.copy()

            if 'Target_encoded' not in data.columns:
                st.error("Kolom Target_encoded tidak ditemukan!")
            else:
                mask = data['Target_encoded'] != 1
                X = data.drop(columns=['Target_encoded'])
                X_filtered = X[mask].reset_index(drop=True)
                y_filtered = data['Target_encoded'][mask].replace([0, 2], [1, 0]).reset_index(drop=True)

                minmax = MinMaxScaler()
                X_scaled = minmax.fit_transform(X_filtered)
                X_scaled_df = pd.DataFrame(X_scaled, columns=X_filtered.columns)

                # Gunakan nama kolom 'Target_encoded' untuk konsistensi
                data_scaled = pd.concat([X_scaled_df, pd.Series(y_filtered, name='Target_encoded')], axis=1)
                st.session_state.data = data_scaled

                st.write("Data Setelah Normalisasi:")
                st.dataframe(data_scaled)

                joblib.dump(minmax, 'model/scaler.pkl')

if selected == "Klasifikasi":
    st.title(f"⏳{selected}")
    data = st.session_state.data.copy()
    X = data.drop(columns='Target_encoded')
    y = data['Target_encoded']

    # Tampilkan opsi rasio
    split_ratio = st.selectbox("Pilih Rasio Pembagian Data", ["90:10", "80:20", "70:30"])

    if split_ratio == "90:10":
        test_size = 0.1
    elif split_ratio == "80:20":
        test_size = 0.2
    else:
        test_size = 0.3

    if len(X) != len(y):
        st.error(f"Panjang fitur (X) dan target (y) tidak sama: {len(X)} vs {len(y)}")
    elif y.nunique() < 2:
        st.error("Target hanya memiliki satu kelas setelah filter. Split dengan stratify tidak bisa dilakukan.")
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        # Tampilkan hasil split
        st.write("Jumlah Data Training :", X_train.shape)
        st.write("Data Training:")
        st.dataframe(pd.concat([X_train, y_train], axis=1))

        st.write("Jumlah Data Testing :", X_test.shape)
        st.write("Data Testing:")
        st.dataframe(pd.concat([X_test, y_test], axis=1))

        # Simpan ke session state
        st.session_state.split_data = (X_train, X_test, y_train, y_test)
    
    if st.button("Modeling Dengan Random Forest"):
        if "split_data" in st.session_state and st.session_state.split_data is not None:
            X_train, X_test, y_train, y_test = st.session_state.split_data
            data = st.session_state.data
            
            parm = {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 10, 15],
                'min_samples_split': [2, 5, 7],
                'min_samples_leaf': [1, 3, 5],
                'max_samples': [0.5, 0.75, 1]
            }

            # Search for best hyperparameters combination
            model = RandomizedSearchCV( estimator = RandomForestClassifier(class_weight = 'balanced', random_state = 42),
                                            param_distributions = parm, scoring = 'balanced_accuracy',
                                            n_iter = 30, n_jobs = -1,  random_state = 0)

            model.fit(X_train, y_train)

            # Get the best estimator
            final_model = model.best_estimator_

            # Check the model performance
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)

            tuned_rf_bi_accuracy = round(balanced_accuracy_score(y_test, y_pred), 3)
            tuned_rf_bi_f1score = round(f1_score(y_test, y_pred), 3)
            tuned_rf_bi_auc = round(roc_auc_score(y_test, y_prob[:, 1]), 3)

            st.write('Performa Model:')
            st.write('Akurasi:', tuned_rf_bi_accuracy)
            st.write('F1 Score:', tuned_rf_bi_f1score)
            st.write('AUC score:', tuned_rf_bi_auc)
            
            joblib.dump(final_model, 'model/randomforest_model.pkl')

if selected == "Prediksi":
    st.title(f"🔍{selected}")
    
    if "data" not in st.session_state:
        st.warning("Data belum tersedia. Silakan lakukan tahapan sebelumnya terlebih dahulu.")
    else:
        data = st.session_state.data
        feature_columns = data.columns[:-1]  # Exclude label/target column

        model = joblib.load("model/randomforest_model.pkl")
        scaler = joblib.load("model/scaler.pkl")

        st.title("Prediksi Status Mahasiswa (Dropout / Tidak)")

        with st.form("form_prediksi"):
            st.subheader("Input Data Mahasiswa")

            # Fitur kategorikal
            marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Widowed','Divorced','Common-law Marriage','Legally Separated'])
            marital_map = {
                'Single': 1, 'Married': 2, 'Widowed': 3,
                'Divorced': 4, 'Common-law Marriage': 5, 'Legally Separated': 6
            }
            
            application_mode = st.selectbox(
                                    "Application Mode", 
                                    [
                                        '1st phase—general contingent',
                                        'Ordinance No. 612/93',
                                        '1st phase—special contingent (Azores Island)',
                                        'Holders of other higher courses',
                                        'Ordinance No. 854-B/99',
                                        'International student (bachelor)',
                                        '1st phase—special contingent (Madeira Island)',
                                        '2nd phase—general contingent',
                                        '3rd phase—general contingent',
                                        'Ordinance No. 533-A/99, item b2) (Different Plan)',
                                        'Ordinance No. 533-A/99, item b3 (Other Institution)',
                                        'Over 23 years old',
                                        'Transfer',
                                        'Change in course',
                                        'Technological specialization diploma holders',
                                        'Change in institution/course',
                                        'Short cycle diploma holders',
                                        'Change in institution/course (International)'])
            application_mode_map = {
                '1st phase—general contingent': 1,
                'Ordinance No. 612/93': 2,
                '1st phase—special contingent (Azores Island)': 3,
                'Holders of other higher courses': 4,
                'Ordinance No. 854-B/99': 5,
                'International student (bachelor)': 6,
                '1st phase—special contingent (Madeira Island)': 7,
                '2nd phase—general contingent': 8,
                '3rd phase—general contingent': 9,
                'Ordinance No. 533-A/99, item b2) (Different Plan)': 10,
                'Ordinance No. 533-A/99, item b3 (Other Institution)': 11,
                'Over 23 years old': 12,
                'Transfer': 13,
                'Change in course': 14,
                'Technological specialization diploma holders': 15,
                'Change in institution/course': 16,
                'Short cycle diploma holders': 17,
                'Change in institution/course (International)': 18
            }
            
            course = st.selectbox(
                            "Course", 
                            [
                                'Biofuel Production Technologies',
                                'Animation and Multimedia Design',
                                'Social Service (evening attendance)',
                                'Agronomy',
                                'Communicatin Design',
                                'Veterinary Nursing',
                                'Informatics Engineering',
                                'Equiniculture',
                                'Management',
                                'Social Service',
                                'Tourism',
                                'Nursing',
                                'Oral Hygiene',
                                'Adversiting and Marketing Management',
                                'Journalism and Communication',
                                'Basic Education',
                                'Management (evening attedance)'])
            course_map = {
                'Biofuel Production Technologies': 1,
                'Animation and Multimedia Design': 2,
                'Social Service (evening attendance)': 3,
                'Agronomy': 4,
                'Communicatin Design': 5,
                'Veterinary Nursing': 6,
                'Informatics Engineering': 7,
                'Equiniculture': 8,
                'Management': 9,
                'Social Service': 10,
                'Tourism': 11,
                'Nursing': 12,
                'Oral Hygiene': 13,
                'Adversiting and Marketing Management': 14,
                'Journalism and Communication': 15,
                'Basic Education': 16,
                'Management (evening attedance)': 17
            }
            
            attendance = st.selectbox("Attendance", ['Evening', 'Daytime'])
            attendance_map = {'Daytime': 1, 'Evening': 0}
            
            prev_qualification = st.selectbox(
                                    "Previous Qualification",
                                    [
                                        'Secondary education',
                                        'Higher education—bachelor’s degree',
                                        'Higher education—degree',
                                        'Higher education—master’s degree',
                                        'Higher education—doctorate',
                                        'Frequency of higher education',
                                        '12th year of schooling—not completed',
                                        '11th year of schooling—not completed',
                                        'Other—11th year of schooling',
                                        '10th year of schooling',
                                        '10th year of schooling—not completed',
                                        'Basic education 3rd cycle (9th/10th/11th year) or equivalent',
                                        'Basic education 2nd cycle (6th/7th/8th year) or equivalent',
                                        'Technological specialization course',
                                        'Higher education—degree (1st cycle)',
                                        'Professional higher technical course',
                                        'Higher education—master’s degree (2nd cycle)'
                                    ]
                                )
            prev_qualification_map = {
                'Secondary education': 1,
                'Higher education—bachelor’s degree': 2,
                'Higher education—degree': 3,
                'Higher education—master’s degree': 4,
                'Higher education—doctorate': 5,
                'Frequency of higher education': 6,
                '12th year of schooling—not completed': 7,
                '11th year of schooling—not completed': 8,
                'Other—11th year of schooling': 9,
                '10th year of schooling': 10,
                '10th year of schooling—not completed': 11,
                'Basic education 3rd cycle (9th/10th/11th year) or equivalent': 12,
                'Basic education 2nd cycle (6th/7th/8th year) or equivalent': 13,
                'Technological specialization course': 14,
                'Higher education—degree (1st cycle)': 15,
                'Professional higher technical course': 16,
                'Higher education—master’s degree (2nd cycle)': 17
            }
            
            mother_qual = st.selectbox(
                            "Mother Qualification",
                            [
                                'Secondary Education—12th Year of Schooling or Equivalent',
                                'Higher Education—bachelor’s degree',
                                'Higher Education—degree',
                                'Higher Education—master’s degree',
                                'Higher Education—doctorate',
                                'Frequency of Higher Education',
                                '12th Year of Schooling—not completed',
                                '11th Year of Schooling—not completed',
                                '7th Year (Old)',
                                'Other—11th Year of Schooling',
                                '2nd year complementary high school course',
                                '10th Year of Schooling',
                                'General commerce course',
                                'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent',
                                'Complementary High School Course',
                                'Technical-professional course',
                                'Complementary High School Course—not concluded',
                                '7th year of schooling',
                                '2nd cycle of the general high school course',
                                '9th Year of Schooling—not completed',
                                '8th year of schooling',
                                'General Course of Administration and Commerce',
                                'Supplementary Accounting and Administration',
                                'Unknown',
                                'Cannot read or write',
                                'Can read without having a 4th year of schooling',
                                'Basic education 1st cycle (4th/5th year) or equivalent',
                                'Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent',
                                'Technological specialization course',
                                'Higher education—degree (1st cycle)',
                                'Specialized higher studies course',
                                'Professional higher technical course',
                                'Higher Education—master’s degree (2nd cycle)',
                                'Higher Education—doctorate (3rd cycle)'
                            ]
                        )
            mother_qual_map = {
                'Secondary Education—12th Year of Schooling or Equivalent': 1,
                'Higher Education—bachelor’s degree': 2,
                'Higher Education—degree': 3,
                'Higher Education—master’s degree': 4,
                'Higher Education—doctorate': 5,
                'Frequency of Higher Education': 6,
                '12th Year of Schooling—not completed': 7,
                '11th Year of Schooling—not completed': 8,
                '7th Year (Old)': 9,
                'Other—11th Year of Schooling': 10,
                '2nd year complementary high school course': 11,
                '10th Year of Schooling': 12,
                'General commerce course': 13,
                'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent': 14,
                'Complementary High School Course': 15,
                'Technical-professional course': 16,
                'Complementary High School Course—not concluded': 17,
                '7th year of schooling': 18,
                '2nd cycle of the general high school course': 19,
                '9th Year of Schooling—not completed': 20,
                '8th year of schooling': 21,
                'General Course of Administration and Commerce': 22,
                'Supplementary Accounting and Administration': 23,
                'Unknown': 24,
                'Cannot read or write': 25,
                'Can read without having a 4th year of schooling': 26,
                'Basic education 1st cycle (4th/5th year) or equivalent': 27,
                'Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent': 28,
                'Technological specialization course': 29,
                'Higher education—degree (1st cycle)': 30,
                'Specialized higher studies course': 31,
                'Professional higher technical course': 32,
                'Higher Education—master’s degree (2nd cycle)': 33,
                'Higher Education—doctorate (3rd cycle)': 34
            }
            
            father_qual = st.selectbox(
                            "Father Qualification",
                            [
                                'Secondary Education—12th Year of Schooling or Equivalent',
                                'Higher Education—bachelor’s degree',
                                'Higher Education—degree',
                                'Higher Education—master’s degree',
                                'Higher Education—doctorate',
                                'Frequency of Higher Education',
                                '12th Year of Schooling—not completed',
                                '11th Year of Schooling—not completed',
                                '7th Year (Old)',
                                'Other—11th Year of Schooling',
                                '2nd year complementary high school course',
                                '10th Year of Schooling',
                                'General commerce course',
                                'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent',
                                'Complementary High School Course',
                                'Technical-professional course',
                                'Complementary High School Course—not concluded',
                                '7th year of schooling',
                                '2nd cycle of the general high school course',
                                '9th Year of Schooling—not completed',
                                '8th year of schooling',
                                'General Course of Administration and Commerce',
                                'Supplementary Accounting and Administration',
                                'Unknown',
                                'Cannot read or write',
                                'Can read without having a 4th year of schooling',
                                'Basic education 1st cycle (4th/5th year) or equivalent',
                                'Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent',
                                'Technological specialization course',
                                'Higher education—degree (1st cycle)',
                                'Specialized higher studies course',
                                'Professional higher technical course',
                                'Higher Education—master’s degree (2nd cycle)',
                                'Higher Education—doctorate (3rd cycle)'
                            ]
                        )
            father_qual_map = {
                'Secondary Education—12th Year of Schooling or Equivalent': 1,
                'Higher Education—bachelor’s degree': 2,
                'Higher Education—degree': 3,
                'Higher Education—master’s degree': 4,
                'Higher Education—doctorate': 5,
                'Frequency of Higher Education': 6,
                '12th Year of Schooling—not completed': 7,
                '11th Year of Schooling—not completed': 8,
                '7th Year (Old)': 9,
                'Other—11th Year of Schooling': 10,
                '2nd year complementary high school course': 11,
                '10th Year of Schooling': 12,
                'General commerce course': 13,
                'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent': 14,
                'Complementary High School Course': 15,
                'Technical-professional course': 16,
                'Complementary High School Course—not concluded': 17,
                '7th year of schooling': 18,
                '2nd cycle of the general high school course': 19,
                '9th Year of Schooling—not completed': 20,
                '8th year of schooling': 21,
                'General Course of Administration and Commerce': 22,
                'Supplementary Accounting and Administration': 23,
                'Unknown': 24,
                'Cannot read or write': 25,
                'Can read without having a 4th year of schooling': 26,
                'Basic education 1st cycle (4th/5th year) or equivalent': 27,
                'Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent': 28,
                'Technological specialization course': 29,
                'Higher education—degree (1st cycle)': 30,
                'Specialized higher studies course': 31,
                'Professional higher technical course': 32,
                'Higher Education—master’s degree (2nd cycle)': 33,
                'Higher Education—doctorate (3rd cycle)': 34
            }
               
            mother_occ = st.selectbox(
                            "Mother Occupation",
                            [
                                "Student",
                                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
                                "Specialists in Intellectual and Scientific Activities",
                                "Intermediate Level Technicians and Professions",
                                "Administrative staff",
                                "Personal Services, Security and Safety Workers, and Sellers",
                                "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry",
                                "Skilled Workers in Industry, Construction, and Craftsmen",
                                "Installation and Machine Operators and Assembly Workers",
                                "Unskilled Workers",
                                "Armed Forces Professions",
                                "Other Situation; 13—(blank)",
                                "Armed Forces Officers",
                                "Armed Forces Sergeants",
                                "Other Armed Forces personnel",
                                "Directors of administrative and commercial services",
                                "Hotel, catering, trade, and other services directors",
                                "Specialists in the physical sciences, mathematics, engineering, and related techniques",
                                "Health professionals",
                                "Teachers",
                                "Specialists in finance, accounting, administrative organization, and public and commercial relations",
                                "Intermediate level science and engineering technicians and professions",
                                "Technicians and professionals of intermediate level of health",
                                "Intermediate level technicians from legal, social, sports, cultural, and similar services",
                                "Information and communication technology technicians",
                                "Office workers, secretaries in general, and data processing operators",
                                "Data, accounting, statistical, financial services, and registry-related operators",
                                "Other administrative support staff",
                                "Personal service workers",
                                "Sellers",
                                "Personal care workers and the like",
                                "Protection and security services personnel",
                                "Market-oriented farmers and skilled agricultural and animal production workers",
                                "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence",
                                "Skilled construction workers and the like, except electricians",
                                "Skilled workers in metallurgy, metalworking, and similar",
                                "Skilled workers in electricity and electronics",
                                "Workers in food processing, woodworking, and clothing and other industries and crafts",
                                "Fixed plant and machine operators",
                                "Assembly workers",
                                "Vehicle drivers and mobile equipment operators",
                                "Unskilled workers in agriculture, animal production, and fisheries and forestry",
                                "Unskilled workers in extractive industry, construction, manufacturing, and transport",
                                "Meal preparation assistants",
                                "Street vendors (except food) and street service provider"
                            ]
                        )
            mother_occ_map = {
                "Student": 1,
                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 2,
                "Specialists in Intellectual and Scientific Activities": 3,
                "Intermediate Level Technicians and Professions": 4,
                "Administrative staff": 5,
                "Personal Services, Security and Safety Workers, and Sellers": 6,
                "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry": 7,
                "Skilled Workers in Industry, Construction, and Craftsmen": 8,
                "Installation and Machine Operators and Assembly Workers": 9,
                "Unskilled Workers": 10,
                "Armed Forces Professions": 11,
                "Other Situation; 13—(blank)": 12,
                "Armed Forces Officers": 13,
                "Armed Forces Sergeants": 14,
                "Other Armed Forces personnel": 15,
                "Directors of administrative and commercial services": 16,
                "Hotel, catering, trade, and other services directors": 17,
                "Specialists in the physical sciences, mathematics, engineering, and related techniques": 18,
                "Health professionals": 19,
                "Teachers": 20,
                "Specialists in finance, accounting, administrative organization, and public and commercial relations": 21,
                "Intermediate level science and engineering technicians and professions": 22,
                "Technicians and professionals of intermediate level of health": 23,
                "Intermediate level technicians from legal, social, sports, cultural, and similar services": 24,
                "Information and communication technology technicians": 25,
                "Office workers, secretaries in general, and data processing operators": 26,
                "Data, accounting, statistical, financial services, and registry-related operators": 27,
                "Other administrative support staff": 28,
                "Personal service workers": 29,
                "Sellers": 30,
                "Personal care workers and the like": 31,
                "Protection and security services personnel": 32,
                "Market-oriented farmers and skilled agricultural and animal production workers": 33,
                "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence": 34,
                "Skilled construction workers and the like, except electricians": 35,
                "Skilled workers in metallurgy, metalworking, and similar": 36,
                "Skilled workers in electricity and electronics": 37,
                "Workers in food processing, woodworking, and clothing and other industries and crafts": 38,
                "Fixed plant and machine operators": 39,
                "Assembly workers": 40,
                "Vehicle drivers and mobile equipment operators": 41,
                "Unskilled workers in agriculture, animal production, and fisheries and forestry": 42,
                "Unskilled workers in extractive industry, construction, manufacturing, and transport": 43,
                "Meal preparation assistants": 44,
                "Street vendors (except food) and street service provider": 45
            }
            
            father_occ = st.selectbox(
                            "Father Occupation",
                            [
                                "Student",
                                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
                                "Specialists in Intellectual and Scientific Activities",
                                "Intermediate Level Technicians and Professions",
                                "Administrative staff",
                                "Personal Services, Security and Safety Workers, and Sellers",
                                "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry",
                                "Skilled Workers in Industry, Construction, and Craftsmen",
                                "Installation and Machine Operators and Assembly Workers",
                                "Unskilled Workers",
                                "Armed Forces Professions",
                                "Other Situation; 13—(blank)",
                                "Armed Forces Officers",
                                "Armed Forces Sergeants",
                                "Other Armed Forces personnel",
                                "Directors of administrative and commercial services",
                                "Hotel, catering, trade, and other services directors",
                                "Specialists in the physical sciences, mathematics, engineering, and related techniques",
                                "Health professionals",
                                "Teachers",
                                "Specialists in finance, accounting, administrative organization, and public and commercial relations",
                                "Intermediate level science and engineering technicians and professions",
                                "Technicians and professionals of intermediate level of health",
                                "Intermediate level technicians from legal, social, sports, cultural, and similar services",
                                "Information and communication technology technicians",
                                "Office workers, secretaries in general, and data processing operators",
                                "Data, accounting, statistical, financial services, and registry-related operators",
                                "Other administrative support staff",
                                "Personal service workers",
                                "Sellers",
                                "Personal care workers and the like",
                                "Protection and security services personnel",
                                "Market-oriented farmers and skilled agricultural and animal production workers",
                                "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence",
                                "Skilled construction workers and the like, except electricians",
                                "Skilled workers in metallurgy, metalworking, and similar",
                                "Skilled workers in electricity and electronics",
                                "Workers in food processing, woodworking, and clothing and other industries and crafts",
                                "Fixed plant and machine operators",
                                "Assembly workers",
                                "Vehicle drivers and mobile equipment operators",
                                "Unskilled workers in agriculture, animal production, and fisheries and forestry",
                                "Unskilled workers in extractive industry, construction, manufacturing, and transport",
                                "Meal preparation assistants",
                                "Street vendors (except food) and street service provider"
                            ]
                        )
            father_occ_map = {
                "Student": 1,
                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 2,
                "Specialists in Intellectual and Scientific Activities": 3,
                "Intermediate Level Technicians and Professions": 4,
                "Administrative staff": 5,
                "Personal Services, Security and Safety Workers, and Sellers": 6,
                "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry": 7,
                "Skilled Workers in Industry, Construction, and Craftsmen": 8,
                "Installation and Machine Operators and Assembly Workers": 9,
                "Unskilled Workers": 10,
                "Armed Forces Professions": 11,
                "Other Situation; 13—(blank)": 12,
                "Armed Forces Officers": 13,
                "Armed Forces Sergeants": 14,
                "Other Armed Forces personnel": 15,
                "Directors of administrative and commercial services": 16,
                "Hotel, catering, trade, and other services directors": 17,
                "Specialists in the physical sciences, mathematics, engineering, and related techniques": 18,
                "Health professionals": 19,
                "Teachers": 20,
                "Specialists in finance, accounting, administrative organization, and public and commercial relations": 21,
                "Intermediate level science and engineering technicians and professions": 22,
                "Technicians and professionals of intermediate level of health": 23,
                "Intermediate level technicians from legal, social, sports, cultural, and similar services": 24,
                "Information and communication technology technicians": 25,
                "Office workers, secretaries in general, and data processing operators": 26,
                "Data, accounting, statistical, financial services, and registry-related operators": 27,
                "Other administrative support staff": 28,
                "Personal service workers": 29,
                "Sellers": 30,
                "Personal care workers and the like": 31,
                "Protection and security services personnel": 32,
                "Market-oriented farmers and skilled agricultural and animal production workers": 33,
                "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence": 34,
                "Skilled construction workers and the like, except electricians": 35,
                "Skilled workers in metallurgy, metalworking, and similar": 36,
                "Skilled workers in electricity and electronics": 37,
                "Workers in food processing, woodworking, and clothing and other industries and crafts": 38,
                "Fixed plant and machine operators": 39,
                "Assembly workers": 40,
                "Vehicle drivers and mobile equipment operators": 41,
                "Unskilled workers in agriculture, animal production, and fisheries and forestry": 42,
                "Unskilled workers in extractive industry, construction, manufacturing, and transport": 43,
                "Meal preparation assistants": 44,
                "Street vendors (except food) and street service provider": 45
            }
            
            displaced = st.selectbox("Displaced", ['Yes', 'No'])
            displaced_map = {'Yes': 1, 'No': 0}
            
            debtor = st.selectbox("Debtor", ['Yes', 'No'])
            debtor_map = {'Yes': 1, 'No': 0}
            
            tuition = st.selectbox("Tuition Fees Up to Date", ['Yes', 'No'])
            tuition_map = {'Yes': 1, 'No': 0}
            
            gender = st.selectbox("Gender", ['Male', 'Female'])
            gender_map = {'Male': 0, 'Female': 1}
            
            scholarship = st.selectbox("Scholarship Holder", ['Yes', 'No'])
            scholarship_map = {'Yes': 1, 'No': 0}

            # Fitur numerik
            application_order = st.number_input("Application Order", 1, 10)
            age = st.number_input("Age", 17, 80)
            credited_1 = st.number_input("Credited 1st Sem", 0.0)
            credited_2 = st.number_input("Credited 2nd Sem", 0.0)
            enrolled_1 = st.number_input("Enrolled 1st Sem", 0.0)
            enrolled_2 = st.number_input("Enrolled 2nd Sem", 0.0)
            eval_1 = st.number_input("Evaluations 1st Sem", 0.0)
            eval_2 = st.number_input("Evaluations 2nd Sem", 0.0)
            approved_1 = st.number_input("Approved 1st Sem", 0.0)
            approved_2 = st.number_input("Approved 2nd Sem", 0.0)
            grade_1 = st.number_input("Grade 1st Sem", 0.0)
            grade_2 = st.number_input("Grade 2nd Sem", 0.0)
            without_eval_1 = st.number_input("Without Evaluations 1st Sem", 0.0)
            without_eval_2 = st.number_input("Without Evaluations 2nd Sem", 0.0)
            gdp = st.number_input("GDP", 0.0)

            submit = st.form_submit_button("Prediksi")

        if submit:
            # Hitung avg fitur
            avg_credited = np.mean([credited_1, credited_2])
            avg_enrolled = np.mean([enrolled_1, enrolled_2])
            avg_eval = np.mean([eval_1, eval_2])
            avg_approved = np.mean([approved_1, approved_2])
            avg_grade = np.mean([grade_1, grade_2])
            avg_wo_eval = np.mean([without_eval_1, without_eval_2])

            # Buat DataFrame input
            input_data = pd.DataFrame([{
                'Marital_status': marital_map[marital_status],
                'Application_mode': application_mode_map[application_mode],
                'Application_order': application_order,
                'Course': course_map[course],
                'Daytime/evening_attendance': attendance_map[attendance],
                'Previous_qualification': prev_qualification_map[prev_qualification],
                'Mother_qualification': mother_qual_map[mother_qual],
                'Father_qualification': father_qual_map[father_qual],
                'Mother_occupation': mother_occ_map[mother_occ],
                'Father_occupation': father_occ_map[father_occ],
                'Displaced': displaced_map[displaced],
                'Debtor': debtor_map[debtor],
                'Tuition_fees_up_to_date': tuition_map[tuition],
                'Gender': gender_map[gender],
                'Scholarship_holder': scholarship_map[scholarship],
                'Age': age,
                'GDP': gdp,
                'avg_enrolled': avg_enrolled,
                'avg_approved': avg_approved,
                'avg_grade': avg_grade,
                'avg_without_evaluations': avg_wo_eval,
            }])

            # Drop fitur yang tidak digunakan
            input_data = input_data.drop(columns=['avg_credited', 'avg_evaluations'], errors='ignore')
            
            # Normalisasi
            input_scaled = scaler.transform(input_data)

            # Prediksi
            pred = model.predict(input_scaled)[0]
            label = "Dropout" if pred == 1 else "Tidak Dropout"

            st.success(f"Hasil Prediksi: **{label}**")