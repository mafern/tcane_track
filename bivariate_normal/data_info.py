

def get_storm_details(df, isample):
    storm = df.iloc[isample]
    storm_name = storm["Name"]
    storm_ftime = storm["ftime(hr)"]
    storm_month = str(storm["time"])[:-4]
    storm_day = str(storm["time"])[-4:-2]
    storm_hour = str(storm["time"])[-2:]
    storm_year = storm["year"]

    details = (
            storm_name
            + " "
            + str(storm_year)
            + "-"
            + str(storm_month)
            + "-"
            + str(storm_day)
            + " "
            + str(storm_hour)
            + "00 @"
            + str(storm_ftime)
            + "hr"
    )

    return details