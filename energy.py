import requests
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pmdarima.arima import ARIMA


class Energy:
    def __init__(self) -> None:
        """
        The first method __init__ serves as the initializer of our class
        Energy. The focus is to retrieve and to make the necessary dataset
        for our analysis assessable for our analysis. The data set in question
        is a csv-file from GitHub. It originated from Our World In Data
        comprising multiple data sources and containing energy data on energy
        consumption (primary energy, per capita, and growth rates), energy mix,
        electricity mix and other relevant metrics.

        Returns
        -------
        None
            It's a constructor.

        """
        url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
        req = requests.get(url)
        url_content = req.content
        csv_file = open('./data.csv', 'wb')
        csv_file.write(url_content)
        csv_file.close()
        self.data = pd.read_csv("data.csv")

    def print_data(self):
        """
        The second method print_data is used to display the data set.

        Returns
        -------
        Pandas.DataFrame()
            First rows of the dataset.

        """
        return self.data.head()

    def clean_data(self):
        """
        The data must now be cleaned so that the dataset can be used for
        further analysis, which will be achieved by the method clean_data.
        First of all, not assigned values will be replaced by zeros.
        Furthermore, the dataset will be reduced to the variable of interest
        “country, year, gdp, renewables_energy_per_capita,
        and fossil_energy_by_capita”.

        “country” contains the geographic location where the observation was
        made
        “year” contains the year of the observation
        “gdp” contains the total real gross domestic product,
        inflation adjusted
        “renewables_energy_per_capita” contains per capita primary energy
        consumption from renewables, measured in kilowatt-hours
        “fossil_energy_by_capita” contains per capita fossil fuel consumption,
        measured in kilowatt-hours. This is the sum of primary energy from
        coal, oil and gas.

        Furthermore, the index of the data table is dropped and set to year.
        The observed time frame for the analysis is comprised to 1970 until
        2018 because the rest of the years contained missing values for a lot
        of countries.


        Returns
        -------
        None.

        """
        self.data = self.data.fillna(0)
        self.data = self.data[[
            "country", "year", "gdp",
            "renewables_energy_per_capita",
            "fossil_energy_per_capita"]].reset_index().drop(["index"], axis=1)
        self.data = self.data[(self.data["year"] >= 1970)
                              & (self.data["year"] <= 2018)]

    def filter_countries(self, countries: list):
        """
        The purpose of the method filter_countries is used to enter
        the countries of interest. Therefore, the method must receive a
        list containing the countries. Afterwards it sets the columns of the
        dataset equal to the countries by transposing the list, then we
        select the corresponding countries, and we transpose once again and
        reset the index.

        Parameters
        ----------
        countries : list
            List of countries.

        Returns
        -------
        df : Pandas.DataFrame()
            Fitlered data.

        """
        df = self.data.set_index("country").T[countries].T.reset_index()
        try:
            df["year"] = pd.DatetimeIndex(
                pd.to_datetime(df["year"], format="%Y")).year
            df = df.set_index("year")
        except KeyError:
            pass

        return df

    def plots(self, data, type: str, countries: list):
        """
        The method plots is used to display the country data over the observed
        time frame. In the first step the size of the plot is choosen to be 16
        by 8. In the next step we check for each country via a for loop, which
        of the three given variables the object receives (previously defined in
        the object clean_data. It can either be “fossil, renewables” or “gdp”.
        Afterwards it plots the respective applicable variable

        Parameters
        ----------
        data : Pandas.DataFrame()
            Data to plot.
        type : str
            Type of variable to plot.
        countries : list
            list of countries.

        Returns
        -------
        Plots.

        """
        plt.figure(figsize=(16, 8))
        for c in countries:
            if type == "renewables":
                plt.plot(data[data["country"] ==
                         c].renewables_energy_per_capita)
                plt.legend(countries)
            elif type == "fossil":
                plt.plot(data[data["country"] == c].fossil_energy_per_capita)
                plt.legend(countries)
            elif type == "gdp":
                plt.plot(data[data["country"] == c].gdp)
                plt.legend(countries)

    def arima_forecast(self, data, countries: list, type: str):
        """
        The arima_forecast function receives a list of the desired countries
        and the variable of interest. In the first step, for each country in
        our list, we check which of the three given types of variables the
        function receives. It can either be “fossil”, “renewables” or “gdp”.
        In the next step, for each type, the same procedure is applied. First,
        we use the auto_arima method to get the best values of the parameters
        used in the ARIMA model, i.e. p, q and d. We save the best combination
        in our variable o and feed this combination to our ARIMA model to
        forecast the next 10 years. Lastly, we plot our past data for each
        country together with our time series prediction from the ARIMA model.

        Parameters
        ----------
        data : Pandas.DataFrame()
            Desired data to forecast and plot.
        countries : list
            Countries list.
        type : str
            Type of variable to plot.

        Returns
        -------
        Plots.

        """
        plt.figure(figsize=(16, 8))
        legend = []
        for c in countries:
            if type == "fossil":
                legend.extend([c, f"Predicted for {c}"])
                best_consumption = auto_arima(
                    data[data["country"] == c].fossil_energy_per_capita,
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
                pred = model_c.fit_predict(
                    y=data[data["country"] == c].fossil_energy_per_capita,
                    x=data[data["country"] == c].index)

                plt.plot(data[data["country"] == c].fossil_energy_per_capita)
                plt.plot(pd.Series(pred, index=[t for t in range(2018, 2028)]))
                plt.legend(legend)

            if type == "renewables":
                legend.extend([c, f"Predicted for {c}"])
                best_consumption = auto_arima(
                    data[data["country"] == c].renewables_energy_per_capita,
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
                pred = model_c.fit_predict(
                    y=data[data["country"] == c].renewables_energy_per_capita,
                    x=data[data["country"] == c].index)

                plt.plot(data[data["country"] ==
                         c].renewables_energy_per_capita)
                plt.plot(pd.Series(pred, index=[t for t in range(2018, 2028)]))
                plt.legend(legend)

            if type == "gdp":
                legend.extend([c, f"Predicted for {c}"])
                best_consumption = auto_arima(
                    data[data["country"] == c].gdp,
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
                pred = model_c.fit_predict(
                    y=data[data["country"] == c].gdp,
                    x=data[data["country"] == c].index)

                plt.plot(data[data["country"] == c].gdp)
                plt.plot(pd.Series(pred, index=[t for t in range(2018, 2028)]))
                plt.legend(legend)
