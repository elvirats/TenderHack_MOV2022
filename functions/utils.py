import pandas as pd


def apply_price_drawdown(x: pd.DataFrame, initial_price: str, final_price: str):
    """Return percent change between Initial and Final prices for method `apply` in pandas"""
    if x[final_price] != 0:
        return 100 * ((x[initial_price] - x[final_price]) / x[initial_price])
    else:
        return 100

def load_classifier_database() -> pd.DataFrame:
    """
    Load the classificator code base with their corresponded names,
    concatenate them and return as the dataframe
    """
    okpd_data = pd.read_excel("codes/okpd.xlsx")
    kpgz_data = pd.read_excel("codes/kpgz.xls")

    kpgz_data.rename(columns={
        'Код КПГЗ': 'Код',
        'Наименование классификации предметов государственного заказа (КПГЗ)': 'Название'
        }, inplace=True)
    return pd.concat([okpd_data[['Код', 'Название']], kpgz_data[['Код', 'Название']]], axis=0)

def find_code_name_in_dict(code: str, code_base: pd.DataFrame) -> str:
    """
    For each code return its name.
    Crop code while the name will be received.
    If there is no code in base, return  empty string"""
    result = ""
    while len(code.split(".")) > 1:
        try:
            # Find name by its code
            result = code_base.loc[code_base['Код'] == code]['Название']
            result = str(result.values[0]).strip()  # strip text
            break
        except:
            # if current code not in base, crop out the last 2 digits
            code = ".".join(code.split(".")[:-1])  # get rid of last sub code
            pass
    return result
    