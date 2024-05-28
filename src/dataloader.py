def simple_analysis(df):
    if isinstance(df, pd.DataFrame):
        print(
            "1 The number of instances of the dataframe is # {}".format(
                df.shape[0]
            ).capitalize()
        )
        print(
            "2 The number of features of the dataframe is # {}".format(
                df.shape[1]
            ).capitalize()
        )

        if df.isnull().sum().sum() > 0:
            print(
                "3.1 The dataframe has missing values and columns with {}".format(
                    list(df.isnull().sum()[df.isnull().sum() > 0])
                )
            )

        else:
            print("3.2# The dataframe has no missing values".capitalize())

        print(
            "4.1 The dataframe has object type columns: {}".format(
                df.dtypes[df.dtypes == "object"].index
            )
        )
        print(
            "4.2 The dataframe has object type columns: {}".format(
                df.dtypes[df.dtypes != "object"].index
            )
        )


if __name__ == "__main__":
    simple_analysis(df)
