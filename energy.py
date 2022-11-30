import requests
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pmdarima.arima import ARIMA

class Energy:
    def __init__(self) -> None:
        url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
        req = requests.get(url)
        url_content = req.content
        csv_file = open('./data.csv', 'wb')
        csv_file.write(url_content)
        csv_file.close()
        self.data = pd.read_csv("data.csv")

    def print_data(self):
        return self.data.head()

    def clean_data(self):
        self.data = self.data.fillna(0)
        self.data = self.data[["country", "year", "gdp", "renewables_energy_per_capita", "fossil_energy_per_capita"]].reset_index().drop(["index"], axis=1)
        self.data = self.data[(self.data["year"] >= 1970)&(self.data["year"] <= 2018)]        


    def filter_countries(self, countries:list):
        df = self.data.set_index("country").T[countries].T.reset_index()
        try:
            df["year"] = pd.DatetimeIndex(pd.to_datetime(df["year"], format="%Y")).year
            df = df.set_index("year")
        except KeyError:
            pass
        
        return df

    def plots(self, data, type:str, countries:list):
        plt.figure(figsize=(16,8))
        for c in countries:
            if type=="renewables":
                plt.plot(data[data["country"]==c].renewables_energy_per_capita)
                plt.legend(countries)
            elif type=="fossil":
                plt.plot(data[data["country"]==c].fossil_energy_per_capita)
                plt.legend(countries)                
            elif type=="gdp":
                plt.plot(data[data["country"]==c].gdp)
                plt.legend(countries)                
      

    def arima_forecast(self, data, countries:list, type:str):
        plt.figure(figsize=(16,8))
        legend = []
        for c in countries:
            if type=="fossil":
                legend.extend([c, f"Predicted for {c}"])
                best_consumption = auto_arima(data[data["country"]==c].fossil_energy_per_capita,
                                            start_p=1,
                                            start_q=1,
                                            max_p=15,
                                            max_q=15,
                                            m=1,
                                            trace=True,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True
                                            )
                o = best_consumption.order
                model_c = ARIMA(order=o)
                pred = model_c.fit_predict(y=data[data["country"]==c].fossil_energy_per_capita, 
                                            x=data[data["country"]==c].index)

                plt.plot(data[data["country"]==c].fossil_energy_per_capita)
                plt.plot(pd.Series(pred, index=[t for t in range(2018,2028)]))
                plt.legend(legend)

            if type=="renewables":
                legend.extend([c, f"Predicted for {c}"])
                best_consumption = auto_arima(data[data["country"]==c].renewables_energy_per_capita,
                                            start_p=1,
                                            start_q=1,
                                            max_p=15,
                                            max_q=15,
                                            m=1,
                                            trace=True,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True
                                            )
                o = best_consumption.order
                model_c = ARIMA(order=o)
                pred = model_c.fit_predict(y=data[data["country"]==c].renewables_energy_per_capita, 
                                            x=data[data["country"]==c].index)

                plt.plot(data[data["country"]==c].renewables_energy_per_capita)
                plt.plot(pd.Series(pred, index=[t for t in range(2018,2028)]))
                plt.legend(legend)

            if type=="gdp":
                legend.extend([c, f"Predicted for {c}"])
                best_consumption = auto_arima(data[data["country"]==c].gdp,
                                            start_p=1,
                                            start_q=1,
                                            max_p=15,
                                            max_q=15,
                                            m=1,
                                            trace=True,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True
                                            )
                o = best_consumption.order
                model_c = ARIMA(order=o)
                pred = model_c.fit_predict(y=data[data["country"]==c].gdp, 
                                            x=data[data["country"]==c].index)

                plt.plot(data[data["country"]==c].gdp)
                plt.plot(pd.Series(pred, index=[t for t in range(2018,2028)]))
                plt.legend(legend)