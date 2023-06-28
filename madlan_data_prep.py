def prepare_data(data):
    
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
    import re
    from sklearn.model_selection import train_test_split
    import statistics
    import string
    from datetime import datetime, timedelta
    import math

    data = data.copy()
     
    def remove_trailing_space(column_names):
        updated_columns = []
        for name in column_names:
            if name.endswith(" "):
                name = name.rstrip()
            updated_columns.append(name)
        return updated_columns
    
    column_names = data.columns.tolist()  # המרת שמות העמודות לרשימה
    updated_column_names = remove_trailing_space(column_names)  # הפעלת הפונקציה על הרשימה
    data.columns = updated_column_names  # הכנסת השמות המעודכנים לDataFrame
    
    data = data.dropna(subset=['price'])
    data = data.drop(data[data['price'] == 'בנה ביתך'].index)
    data = data.drop(data[data['price'] == 'nan'].index)
    data = data.drop_duplicates()
    
    data['City'] = data['City'].replace(['נהרייה', ' נהריה', ' נהרייה'], 'נהריה') #אולי צריך להוסיף טרי כי אולי בדאטא שלהם לא יהיה ואז תיהיה שגיאה
    data['City'] = data['City'].replace(' שוהם', 'שוהם') #כנל

        
    #price
    def regex (price):
        if ',' in str(price):
            price = price.replace(',','')
        if not price.isnumeric():
            numbers = re.findall('[0-9]+', str(price))
            if numbers: 
                price = int(numbers[0]) #למה int ולא float?
        else:
            price = int(price)
        return price
    data['price'] = data['price'].astype(str).apply(regex)
    
    # חילוץ מספר מתוך שטח
    def extract_numbers(text):
        pattern = r'\d+\.?\d*'
        numbers = re.findall(pattern, text)
        if len(numbers) > 0 :
            return float(numbers[0])
        else:
            return np.nan
    data['Area'] = data['Area'].astype(str).apply(extract_numbers)
    data['room_number'] = data['room_number'].astype(str).apply(extract_numbers)

    
    # פונצקיות להורדת סימני פיסוק ואימוג'ים
    def remove_punc(text):
        translator = str.maketrans('', '', string.punctuation) # הסימנים !@#$%^&*(){}[]:;<>,.?/~`-
        clean_text = text.translate(translator)
        clean_text = clean_text.replace("׳", "")
        clean_text = clean_text.replace('״', '')
        return clean_text
    
    #מחיקת אימוגים
    def remove_emojis(text):
        pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # אימוג'י
                             u"\U0001F300-\U0001F5FF"  # סמלים
                             u"\U0001F680-\U0001F6FF"  # סמלי תחבורה ומפות
                             u"\U0001F1E0-\U0001F1FF"  # דגלים
                             u"\U000024C2-\U0001F251"  # תווים נוספים
                             "]+", flags=re.UNICODE)
        return pattern.sub(r'', text)

    data[['Street', 'city_area', 'publishedDays', 'description']] = data[['Street', 'city_area', 'publishedDays', 'description']].astype(str).apply(lambda x: x.apply(remove_punc).apply(remove_emojis))
    data['type'] = data.apply(lambda row: remove_punc(row['type']), axis=1)


    #תאריך כניסה
    def classify_entrance_date(date):
        today = datetime.today()  # תאריך היום
        if isinstance(date, str):
            if date == 'מיידי':
                return 'less_than_6 months'
            elif date == 'גמיש':
                return 'flexible'
            elif date == 'לא צויין':
                return 'not_defined'
            else:
                try:
                    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return 'not_defined'

        months_diff = (date.year - today.year) * 12 + (date.month - today.month)
        days_diff = (date - today).days
        if months_diff < 6:
            return 'less_than_6 months'
        elif months_diff < 12:
            if months_diff == 6 and days_diff < 30*6: # ממוצע, אם בא לך את יכולה לעשות דלטא-טיים
                return 'less_than_6 months'
            else:
                return 'months_6_12'
        else:
            if months_diff == 12 and days_diff < 365: # יש פער של פחות מחודש
                return 'months_6_12'
            else:
                return 'above_year'

    data['entrance_date'] = data['entranceDate'].apply(classify_entrance_date) # זמן כניסה

    # חילוץ הקומה
    #אולי בעייתי ההתייחסות לנאנים גם בטוטאל
    def extract_floor(text):
        floor = re.findall(r'\d+', text) # נחזיר את כל המספרים הקיימים בטקסט
        if floor: # אם יש מספרים- נחזיר את המספר הראשון
            return int(floor[0])
        if text == 'nan': 
            return 1 #למה 1 ולא 0
        elif text == 'קומת מרתף':
            return -1
        else: # למשל קומת קרקע
            return 0

    # חילוץ מספר הקומות בטוטל
    def extract_total_floors(text):
        floor = re.findall(r'\d+', text) # נחזיר את כל המספרים הקיימים בטקסט
        if len(floor) >= 2: # אם יש מספרים- נחזיר את המספר הראשון
            return floor[1]
        elif floor:
            return floor[0]
        else:
            return 1

    data['floor'] = data['floor_out_of'].astype(str).apply(extract_floor)
    data['total_floors'] = data['floor_out_of'].astype(str).apply(extract_total_floors).astype(float)
        
    # הפונקציה תהפוך את כל הערכים הרלוונטיים ל1 (ערך חיובי), את השאר- לשלילה
    def replace_column_values(column_name):
        b= ['יש', 'yes','True','כן']
        list1 = str(column_name).split()
        if list1[0] in b:
            value = 1
        else:
            value = 0
        return value
    
    
    columns1 = ['hasElevator','hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly']
    data[columns1] = data[columns1].applymap(replace_column_values)
    
    
    data = data.replace(['nan','Nan', 'None', 'חדש', 'N/A', 'null', '', False], np.nan)
    data['room_number'] = data['room_number'].astype(str).apply(extract_numbers)
    
    def filter_letters_with_spaces(text):
        # מחיקת תווים שאינם אותיות או רווחים
        letters_only = re.sub("[^א-ת ]", "", text)
    
        return letters_only
    
    def remove_whitespace(sentence):
        if len(sentence) > 0 and sentence[0] == ' ':
            sentence = sentence[1:]
        if len(sentence) > 0 and sentence[-1] == ' ':
            sentence = sentence[:-1]
        return sentence
    
    data['city_area'] = data['city_area'].astype(str).apply(remove_whitespace)
    data['Street'] = data['Street'].astype(str).apply(filter_letters_with_spaces)
    data['Street'] = data['Street'].apply(remove_whitespace)
    data['publishedDays'] = data['publishedDays'].astype(str).apply(remove_whitespace) # אולי להפעיל על כל העמודות?
    data = data.replace(['nan','Nan', 'None', 'חדש', 'N/A', 'null', '', False ,'-'], np.nan)
    data['publishedDays'] = data['publishedDays'].astype(float)
    data['number_in_street'] = data['number_in_street'].astype(float)# צריך להיות int
    data = data.drop('entranceDate', axis=1) # לא צריך את העמודה המקורית
    
    
    def fill_missing_area(row):
        room_number = row['room_number']
        #property_type = row['type']
        area = row['Area']
        if pd.isna(area):
            area =  data[data['room_number'] == room_number]['Area'].tolist() #(data['type'] == property_type) &
            area1 = [x for x in area if not math.isnan(x)]
            if len(area1) > 0: # אם מצאנו ערכים לפי מה שהגדרנו
                vals = []
                for val in area1:
                    vals.append(val)
                if len(vals) > 0: 
                    area = statistics.mean(vals)
            else:
                area =  data.dropna(subset=['Area'])['Area'].mean()
        return float(area)
    
    data['Area'] = data.apply(fill_missing_area, axis=1)
    
        
            
    def fix_publishedDays(val):
        if pd.isnull(val):
            val = 1.0
        return float(val)
    
    data['publishedDays'] = data['publishedDays'].apply(fix_publishedDays)#.astype(float)
    
    def fill_missing_room(row):
        room_number = row['room_number']
        #property_type = row.get['type'] בגלל שהעמודה הזאת לא נומרית הוא לא מכיר אותה בפייפלין שבניתי
        area = row['Area']
        if pd.isna(room_number): # שמה את זה בהערה כי נראלי לא נכון להייחס לרעש גם ככה הדאטא קטןor room_number == '' or float(room_number) > 10: # בהנחה שאף אחד לא מוכר טירה במדלן...
            rooms =  data[data['Area'] == area]['room_number'].tolist()
            rooms1 = [x for x in rooms if not pd.isna(x)]
            if len(rooms1) > 0: # אם מצאנו ערכים לפי מה שהגדרנו
                vals = []
                for val in rooms1:
                    if isinstance(val, (int, float)):
                        # אוולי מיותר val = float(val)
                        vals.append(val)
                if len(vals) > 0: 
                    room_number = statistics.mean(vals)
            else:
                room_number =  data.dropna(subset=['room_number'])['room_number'].mean()

        return float(room_number)

    data['room_number'] = data.apply(fill_missing_room, axis=1)
    
    
    data['num_of_images'] = np.nan_to_num(data['num_of_images'], nan=0)


    def fill_missing_condition(condition):
        if pd.isnull(condition):
            condition = 'לא צויין'
            return condition
        else:
            return condition
                
    data['condition'] = data['condition'].apply(fill_missing_condition) # מצב

    
    def fill_common_city_area(row):
        city = row['City']
        city_area = row['city_area']
        if pd.isna(city_area) or isinstance(city_area, int):
            common_city_area = data[data['City'] == city]['city_area'] # הערך השכיח ביותר באזורי העיר עבור העיר הנוכחית.
            common_city_area1 = [x for x in common_city_area if not pd.isna(x)]
            if len(common_city_area1) >= 1: # אם יש כמה ערכים נפוצים באותה המידה- נבחר את הראשון מביניהם
                common_city_area =  statistics.mode(common_city_area1)
                if len(common_city_area) > 1:
                    common_city_area = common_city_area[0]
            else: # במקרה קיצון שהכל ריק והוא לא מצא כלום
                common_city_area = statistics.mode(data.dropna(subset=['city_area'])['city_area'])#.mode()
            return common_city_area
        else:
            return city_area
        
    data['city_area'] = data.apply(fill_common_city_area, axis=1)
    data = data.drop('floor_out_of', axis=1) # לא צריך את העמודה המקורית
    data = data.drop('number_in_street', axis=1) # יש לנו יותר מידי חסרים (כרבע מהנתונים)

    
    def fill_nan_street_name(row):
        street_name = row['Street']
        city_area = row['city_area']
        city = row['City']
        if pd.isna(street_name):
            street_name = data[(data['city_area']== city_area) & (data['City']== city) ]['Street'].tolist() #[X_df['city_area']== city_area &
            street_name1 = [x for x in street_name if x != 'nan']
            if len(street_name1) >= 1:
                street = statistics.mode(street_name1)
                    #if len(street)>1: מחזיר שכיח אחד
                     #   street = street[0]
                #elif len(street_name1) == 1:
                 #   street = street_name1[0]
            else: # במקרה קיצון שהכל ריק והוא לא מצא כלום
                street = 'לא צויין'#data.dropna(subset=['Street'])['Street'].mode()
            return street
        else:
            return street_name

    data['Street'] = data.apply(fill_nan_street_name, axis=1)
    
    def fill_nan_description(description):
        if pd.isna(description): 
            return 'לא צויין'
        else:
            return description             

    data['description'] = data['description'].apply(fill_nan_description).astype(str)
    data['Street'] = data['Street'].apply(fill_nan_description).astype(str)
    data = data.drop('total_floors', axis=1) # מצאתי קורולציה מאוד גבוהה בין טוטאל למספר קומה אז מחקתי


       
    #column_transformer.fit(data)
    #data = column_transformer.transform(data)
    return data

