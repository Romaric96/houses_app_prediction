###########################################################################################################
###########################################################################################################
###########################################################################################################
######## Authors = NZEUGANG NGOMSEU Romaric & DJEUNANG KENFACK Michèle
######## Insitution = INSTITUT SAINT JEAN - CAMEROON
######## Website = https://institutsaintjean.org/
######## version = 1.0
######## status = PROJET TUTORE
######## Year = 2022/2023
######## deployed at = https://share.streamlit.io/Houses.py
######## Mentor = Dr. SIYOU Vanel
###########################################################################################################
###########################################################################################################
###########################################################################################################

### Import important librairies to be used in our Houses prediction app
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
import plotly.express as px


# Set wide view
st.set_page_config(layout="wide")

### Data Import ###
df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")
types = ["Mean","Absolute","Median","Maximum","Minimum"]

### SIDE BAR ###
sidebar = st.sidebar.container()
with sidebar:
    st.image('icons8-home-188.png', width=200)
    st.sidebar.header('HOUSE PRICE PREDICTION')

########################
### ANALYSIS METHODS ###
########################
# Missing values functions
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)      
        # Coumn for dtypes
        dtype = df.dtypes
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent,dtype], axis=1)
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2:'Data Types'})
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Handling categorical variables
def freq_imp(df,variable):
    frq_cat = df[variable].mode()[0]
    df[variable].fillna(frq_cat, inplace=True)

# Handling Numerical variables
def freq_imp(df,variable):
    frq_num = df[variable].median()
    df[variable].fillna(frq_num, inplace=True)

# Drop missing values cols with more than 50 percent of data
def drop_missing_cols(threshold, df):
    missing_values = pd.DataFrame({'Column': df.columns,
                                   'MissingValues': df.isnull().sum(),
                                   '% of Total Values': df.isnull().sum() / len(df) * 100})
    missing_values = missing_values.sort_values('% of Total Values', ascending=False).reset_index(drop=True)
    
    drop_cols = missing_values[missing_values['% of Total Values'] > threshold]['Column'].tolist()
    df = df.drop(columns=drop_cols)
    
    remaining_missing_values = pd.DataFrame({'Column': df.columns,
                                              'MissingValues': df.isnull().sum(),
                                              '% of Total Values': df.isnull().sum() / len(df) * 100})
    remaining_missing_values = remaining_missing_values.sort_values('% of Total Values', ascending=False).reset_index(drop=True)
    
    return df, remaining_missing_values

# Imputation function
def freq_impute(df, threshold, cols_with_missing):
    # drop columns with missing values above threshold
    drop_cols = df[df['percent of Total Values'] > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    
    # perform frequency imputation for categorical variables with missing values
    cat_cols = [col for col in cols_with_missing if df[col].dtype == 'object']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # perform frequency imputation for numerical variables with missing values
    num_cols = [col for col in cols_with_missing if df[col].dtype == 'float64']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df

# Outlier detection
def plot_boxplots2(dataframe):
    num_features = dataframe.select_dtypes(include='number').columns
    num_cols = len(num_features)
    plt.figure(figsize=(16, 2*num_cols))
    plotnumber = 1
    for col in num_features:
        if plotnumber <= num_cols:
            ax = plt.subplot(num_cols, 8, plotnumber)
            sns.boxplot(dataframe[col], color='green')
            plt.xlabel(col, fontsize=10)
        plotnumber += 1
    plt.tight_layout()
    plt.show()
    st.pyplot()

# Visualization plots
def plot_zoning_classification(df):
    # Pie chart
    labels = df["MSZoning"].unique()
    sizes = df["MSZoning"].value_counts().values
    explode=[0.1,0,0,0,0]
    parcent = 100.*sizes/sizes.sum()
    labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]
    colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']
    patches, texts= plt.pie(sizes, colors=colors,explode=explode,
                            shadow=True,startangle=90)
    plt.legend(patches, labels, loc="best", fontsize=16)
    plt.title("Zoning Classification", fontsize=15)
    plt.gcf().set_size_inches(10, 10)  # Set the figure size
    st.pyplot()
    # Violin plot
    fig, ax = plt.subplots()
    sns.violinplot(x=df.MSZoning,y=df["SalePrice"], ax=ax)
    plt.title("MSZoning wrt Sale Price", fontsize=13)
    plt.xlabel("MSZoning", fontsize=13)
    plt.ylabel("Sale Price", fontsize=13)
    plt.gcf().set_size_inches(10, 6)  # Set the figure size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(fig)

# Scatter plot
def scatter_plot(df, x_col, y_col, title, x_label, y_label, color):
    # set the style to "darkgrid"
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, color=color, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel(x_label, fontsize=13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(fig)

def scatter_plot2(df, x_var, y_var, x_label, y_label, title, color):
    df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
    df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
    plt.scatter(df[x_var], df[y_var], color=color)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.title(title, fontsize=13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    st.pyplot()

# Histograms
def plot_sale_price_per_sqft(df, color):
    df['SalePriceSF'] = df['SalePrice']/df['GrLivArea']
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePriceSF'], bins=15, color=color)
    plt.title("Sale Price per Square Foot", fontsize=13)
    plt.ylabel('Number of Sales', fontsize=13)
    plt.xlabel('Price per square feet', fontsize=13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot()

def plot_sale_price_sqft(df):
    df['SalePriceSF'] = df['SalePrice']/df['GrLivArea']
    plt.hist(df['SalePriceSF'], bins=15, color="gold")
    plt.title("Sale Price per Square Foot", fontsize=13)
    plt.ylabel('Number of Sales', fontsize=13)
    plt.xlabel('Price per square feet', fontsize=13)
    st.pyplot()

#Boxplot
def plot_boxplot(df, x_col, y_col, title):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(title, fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    st.pyplot()

def rooms_vs_saleprice(dataframe):
    sns.barplot(x=dataframe["TotRmsAbvGrd"], y=dataframe["SalePrice"], palette="Blues_d")
    plt.title("Sale Price vs Number of rooms", fontsize=13)
    plt.xlabel("Number of rooms above ground", fontsize=13)
    plt.ylabel("Sale Price (in dollars)", fontsize=13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    st.pyplot()

def plot_neighborhood_sale_price(df, fontsize=12):
    plt.figure(figsize=(16, 6))
    plt.xticks(rotation=45, fontsize=fontsize)
    sns.barplot(x=df["Neighborhood"], y=df["SalePrice"], palette="Blues_d")
    plt.title("Sale Price vs Neighborhood", fontsize=fontsize+4)
    plt.xlabel("Neighborhood", fontsize=fontsize+2)
    plt.ylabel("Sale Price", fontsize=fontsize+2)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    st.pyplot()

def plot_horizontal_bar(x, y, xlabel, ylabel, title, color='r', figsize=(10,6), fontsize=12):
    plt.figure(figsize=figsize)
    plt.barh(x, width=y, color=color)
    plt.title(title, fontsize=fontsize+2)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    st.pyplot()


####################
### INTRODUCTION ###
####################
def home_page():
    # Load the image and apply a blur effect
    img = Image.open("houses.jpg")
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=0))
    # Display the blurred image
    st.image(blurred_img, use_column_width=True)

    option1 = st.sidebar.selectbox(
    'Choose the DATA to visualize',
    ('Train Dataset', 'Test Dataset') 
    )
    option3 = st.sidebar.selectbox(
    'Choose the Description to visualize',
    ('Data Description', 'Field Description') 
    )
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title('PROJECT 5: PREDICTING THE SALE PRICE OF A HOUSE')
    with row0_2:
        st.text("")
        st.subheader('NGOMSEU & DJEUNANG from [Institut Saint Jean](https://institutsaintjean.org/)')
        # Add a container for the logo
        header = st.container()
        # Add the logo to the container
        with header:
            st.image('ISJ.jpeg', width=350)

    row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
    with row3_1:
        st.subheader("I - CONTEXT:")
        st.markdown("Ask a buyer to describe their dream home and they probably won't start with basement ceiling height or proximity to an east-west railroad track. But the data set from this playground competition proves that price negotiations are influenced by much more than the number of bedrooms or a white picket fence.")
        st.subheader("II - PRACTICAL SKILLS")
        st.markdown("Creative feature engineering Advanced regression techniques like random forest and gradient boosting.")
        st.markdown("With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this contest challenges you to predict the final price of each home.")
        st.subheader("III - ACKNOWLEDGEMENTS")
        st.markdown("The Ames housing dataset was compiled by Dean De Cock for use in teaching Data Science. It is an incredible alternative for data scientists looking for a modernized and extended version of the oft-cited Boston housing dataset.")
        st.markdown("We are trying to predict for how much money each house can be sold??, In this mainly we will look at data exploration and visulisation part")
        st.markdown("Exploratory Data Analysis is often most tedious and boring job, But the more time you spend here on understanding, cleaning and preparing data the better fruits your predictive model will bare!!")


    ### SEE DATA ###
    row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
    with row4_1:
        st.subheader("CURRENTLY SELECTED DATA:")

    row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
    if option1 == 'Train Dataset':
        with row3_1:
            st.markdown("")
            see_data = st.expander('You can click here to see the raw data first 👉')
            with see_data:
                st.dataframe(data=df_train.reset_index(drop=True))
    else:
        with row3_1:
            st.markdown("")
            see_data = st.expander('You can click here to see the raw data first 👉')
            with see_data:
                st.dataframe(data=df_test.reset_index(drop=True))

    if option3 == 'Data Description':
        with row3_1:
            st.subheader("DESCRIPTION DES VARIABLES:")
            st.markdown("")
            see_data = st.expander('You can click here to see the file 👉')
            with see_data:
                file_path = "data_description.txt"
                with open(file_path, "r") as f:
                    file_contents = f.read()
                    st.write(file_contents)    
    else:
        with row3_1:
            st.subheader("DESCRIPTION DES VARIABLES:")
            st.markdown("")
            see_data = st.expander('You can click here to see the file 👉')
            with see_data:
                file_path = "datafield.txt"
                with open(file_path, "r") as f:
                    file_contents = f.read()
                    st.write(file_contents)
        
    with row3_1:
        st.text("")
        st.subheader('Link to data description in kaggle [Click here to get redirected](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)')


def page1():
    # Load the image and apply a blur effect
    img = Image.open("exploratory.jpg")
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=0))
    # Display the blurred image
    st.image(blurred_img, use_column_width=True)
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.001, 20, .01, .3, .1))
    with row0_1:
        st.title('**EXPLORATORY ANALYSIS FOR HOUSE PREDICTION**')
        st.markdown('')
        st.subheader("**I - INTRODUCTION**")
        st.markdown('')
        st.markdown("**I - i) Description Statistics**")
        st.write(df_train.describe())

        row11_1, row11_2 = st.columns((1.3, 2))
        with row11_1:
            st.markdown('**I - ii) Descriptive statistics summary for the target only**')
            st.write(df_train['SalePrice'].describe())
            st.markdown('Very well... It seems that your minimum price is larger than zero. We have an average price of 163000 and a maximum selling price of 755000 of the 1460 individuals.')
            st.markdown("**Excellent!**")
        with row11_2:
            st.markdown('**I - iii) Histogram to show the distribution on the sale price**')
            fig = px.histogram(df_train, x='SalePrice', nbins=20)
            st.plotly_chart(fig)
            st.markdown("Deviate from the normal distribution. Have appreciable positive skewness. Show peakedness.")
        st.markdown('')
        st.subheader('**II - CORRELATION**')
        st.markdown('')
        row11_3, row11_4 = st.columns((1, 2))
        with row11_3:
            st.markdown('**II - i) Correlation between variables**')
            corr = df_train.corr()["SalePrice"]
            st.write(corr[np.argsort(corr, axis=0)[::-1]])
            st.markdown('OverallQual ,GrLivArea ,GarageCars,GarageArea ,TotalBsmtSF, 1stFlrSF ,FullBath,TotRmsAbvGrd have more than 0.5 correlation with SalePrice.')
            st.markdown("EnclosedPorch and KitchenAbvGr have little negative correlation with target variable. These can prove to be important features to predict SalePrice.")
        with row11_4:
            st.markdown('**II - ii) Hitmap to show correlation**')
            sns.set(style="whitegrid", color_codes=True)
            sns.set(font_scale=0.5)
            corrMatrix=df_train[["SalePrice","GrLivArea","GarageCars",
                            "GarageArea","TotalBsmtSF","1stFlrSF","FullBath",
                            "TotRmsAbvGrd"]].corr()
            sns.set(font_scale=0.5)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
                        square=True,annot=True,cmap='viridis',linecolor="white", ax=ax)
            plt.title('Correlation between features', fontsize=12)
            st.pyplot(fig)
            
        st.markdown('')
        st.subheader("III - MISSING VALUES INPUTATION")
        st.markdown('')
        st.markdown('But filling missing values with mean/median/mode or using another predictive model to predict missing values is also a prediction which may not be 100 accurate, instead you can use models like Decision Trees and Random Forest which handle missing values very well.')
        st.markdown('Some of this part is based on this kernel [Click here](https://www.kaggle.com/bisaria/house-prices-advanced-regression-techniques/handling-missing-data)')
        st.markdown('')
        st.markdown('**III - i) Visualising missing values in a plot**')
        # Visualising in a plot
        fig, ax = plt.subplots(figsize=(20,5))
        sns.heatmap(df_train.isnull(), cbar=False, cmap="YlGnBu_r")
        plt.show()
        st.pyplot(fig)
        row11_5, row11_6, row11_8, row11_7 = st.columns((5, 1, 1, 5))
        with row11_5:
            st.markdown('**III - ii) Figure out missing value columns**')
            missing_val = missing_values_table(df_train)
            st.write(missing_val)
        st.markdown('We see here in column PoolQC 99.5 % values are missing, Not enough data to take insight from him. MiscFeature, Alley this column also lot of missing values. So, we can decide 80 threshold to delete columns if column have more than 80% data missing we simply drop those columns')
        st.markdown('**Describing Categorical and Numerical features separately after handling the missing values from our data**')
        with row11_6:
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('')
            img = Image.open("arrow.png")
            st.image(img, use_column_width=True)
        with row11_7:
            threshold = 80
            drop_col, house_missing = drop_missing_cols(threshold, df_train)
            st.markdown('III - iii) Missing Values Table after removing some columns')
            st.write(house_missing)
        row11_9, row11_10 = st.columns((5, 3))
        with row11_9:
            st.markdown('III - iv) Describing Categorical')
            num_features = df_train.select_dtypes(include=['int64','float64'])
            st.write(num_features.describe().T)
        with row11_10:
            st.markdown('III - v) Describing Categorical')
            categorical_features = df_train.select_dtypes(include='object')
            st.write(categorical_features.describe().T)
        
        st.markdown('')
        st.subheader("IV - OUTLIERS DETECTION")
        st.markdown('')
        st.write(plot_boxplots2(df_train))

        st.markdown('')
        st.subheader("V - VISUALIZATION")
        st.markdown('')
        row11_11, row11_12 = st.columns((5, 5))
        st.markdown('')
        with row11_11:
            st.markdown('V - i) MSZoning')
            plot_zoning_classification(df_train)
            st.markdown('V - iv) Price per square feet')
            plot_sale_price_per_sqft(df_train, color="yellow")
            st.markdown('V - vi) Total rooms above grade')
            rooms_vs_saleprice(df_train)
            st.markdown('V - viii) Neighbourhood')
            plot_neighborhood_sale_price(df_train)
            st.markdown('V - ix) Overall Quality')
            plot_horizontal_bar(df_train["OverallQual"], df_train["SalePrice"], "Sale Price", "Overall Quality of house", "Sale Price vs Overall Quality of house", color='r', figsize=(10,6), fontsize=12)

        with row11_12:
            st.markdown('V - ii) 1st Floor in square feet')
            scatter_plot(df_train, "1stFlrSF", "SalePrice", "Sale Price wrt 1st floor", "1st Floor in square feet", "Sale Price (in dollars)", "red")
            st.markdown('V - iii) Ground living Area')
            scatter_plot(df_train, "GrLivArea", "SalePrice", "Sale Price wrt Ground living area", "Ground living area", "Sale Price (in dollars)", "purple")
            st.markdown('V - v) Garage Area')
            scatter_plot(df_train, "GarageArea", "SalePrice", "Sale Price vs Garage Area", "Garage Area in sq foot", "Sale Price (in dollars)", "green")
            st.markdown('V - vii) Bathrooms in house')
            plot_boxplot(df_train, "FullBath", "SalePrice", "Sale Price vs Full Bathrooms")
            st.markdown('V - x) 2nd Floor with SalePrice')
            scatter_plot(df_train, "2ndFlrSF", "SalePrice", "Sale Price vs 2nd floor in sq feet", "2nd floor in sq feet", "Sale Price (in dollars)", "gold")
            st.markdown('')
            st.markdown('')
        st.title('**END OF EXPLORATION**')
            



def page2():
    # Load the image and apply a blur effect
    img = Image.open("ISJ.jpg")
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=0))
    # Display the blurred image
    st.image(blurred_img, use_column_width=True)

    # Set page layout
    col1, col2 = st.columns((3, 3))
    with col1:
        st.image("ISJ.jpeg.", width=500)
        st.write("## NZEUGANG NGOMSEU")
        st.write("### Romaric")
        st.write("Field: " + "Data Science & AI")
        st.write("Level: " + "Masters 2")
        st.write("University: INSTITUT SAINT JEAN")
        st.write("Description: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ")

    with col2:
        st.image("ISJ.jpeg.", width=500)
        st.write("## DJEUNANG KENFACK")
        st.write("### Aude Michèle")
        st.write("Field: " + "Data Science & AI")
        st.write("Level: " + "Masters 2")
        st.write("University: INSTITUT SAINT JEAN")
        st.write("Description: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ")

pages = {
    'Home': home_page,
    'Page 1': page1,
    'Page 2': page2
}

# Define the sidebar buttons
with st.sidebar.container():
    st.markdown('## MENU')
    button_1 = st.sidebar.button('**Home Page**')
    button_2 = st.sidebar.button('**Exploratory**')
    button_3 = st.sidebar.button('**About Us**')
    # Use CSS to align the buttons in the sidebar
    st.markdown(
        """
        <style>
        .sidebar .widget-button {
            width: 150px;
            height: 50px;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
 
# Display the appropriate page based on the button clicked
if button_1:
    home_page()
elif button_2:
    page1()
elif button_3:
    page2()
else:
    home_page()