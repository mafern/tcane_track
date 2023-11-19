def get_storm_details(df, isample):
    storm = df.iloc[isample]
    storm_name = storm["NAME"]
    storm_ftime = int(storm["FHOUR"])
    storm_month = str(int(storm["MMDDHH"]))[:-4]
    storm_day = str(int(storm["MMDDHH"]))[-4:-2]
    storm_hour = str(int(storm["MMDDHH"]))[-2:]
    storm_year = int(storm["YEAR"])

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
