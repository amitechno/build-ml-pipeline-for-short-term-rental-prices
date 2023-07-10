import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """
    Test if the columns in the provided data match the expected column names.

    Args:
        data (pd.DataFrame): Input DataFrame to be tested.

    Raises:
        AssertionError: If the columns in the data do not match the expected column names.
    """

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):
    """
    Test if the unique values in the 'neighbourhood_group' column match the known neighborhood names.

    Args:
        data (pd.DataFrame): Input DataFrame to be tested.

    Raises:
        AssertionError: If the unique values in the 'neighbourhood_group' column do not match the known names.
    """

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test if the latitude and longitude values in the data fall within the proper boundaries for properties in and around NYC.

    Args:
        data (pd.DataFrame): Input DataFrame to be tested.

    Raises:
        AssertionError: If any latitude or longitude values fall outside the specified boundaries.
    """
    idx = data['longitude'].between(-74.25, - \
                                    73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Test if the distribution of the 'neighbourhood_group' column in the new data is significantly similar to the reference dataset.

    Args:
        data (pd.DataFrame): New data to be tested.
        ref_data (pd.DataFrame): Reference dataset for comparison.
        kl_threshold (float): Threshold value for the Kullback-Leibler divergence.

    Raises:
        AssertionError: If the KL divergence between the distributions exceeds the specified threshold.
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    """
    Test if the number of rows in the data falls within the expected range.

    Args:
        data (pd.DataFrame): Input DataFrame to be tested.

    Raises:
        AssertionError: If the number of rows in the data is outside the expected range.
    """
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    """
    Test if all the prices in the 'price' column of the data fall within the specified range.

    Args:
        data (pd.DataFrame): Input DataFrame to be tested.
        min_price (float): Minimum price value.
        max_price (float): Maximum price value.

    Raises:
        AssertionError: If any price values fall outside the specified range.
    """
    assert data['price'].between(min_price, max_price).all()
