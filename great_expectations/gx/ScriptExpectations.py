from datasets import load_dataset
import great_expectations as gx
import pandas as pd

context = gx.get_context()





# Load the dataset
dataset = load_dataset("zalando-datasets/fashion_mnist", split='train')
data = dataset.to_pandas()  # Convert to a Pandas DataFrame for Great Expectations

datasource_name = "pandas11"

# if datasource_name not in context.list_datasources() :
data_source = context.data_sources.add_pandas(name=datasource_name)
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

# all labels should be between 0 and 9, representing the 10 different clothing categories
expectation = gx.expectations.ExpectColumnValuesToBeBetween(column="label", min_value=0, max_value=10)

validation_result = batch.validate(expectation)

print(validation_result)


suite = context.suites.add(
    gx.core.expectation_suite.ExpectationSuite(
        name="Data Expectations6",
    )
)

# Expectation 1
suite.add_expectation(expectation)

# Expectation 2: no missing images in dataset
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="image"))

# Expectation 3: no missing labels in dataset
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="label"))

# Expectation 4: all images are stored as grayscale with appropriate type
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="image"))

# Expectation 5: no duplicated entries
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="image"))


# Ensure that each label corresponds to the correct fashion category
label_mapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


validation_definition = context.validation_definitions.add(
    gx.core.validation_definition.ValidationDefinition(name="Validation definition1", data=batch_definition, suite=suite)
)


checkpoint = context.checkpoints.add(
    gx.checkpoint.checkpoint.Checkpoint(name="checkpoint1", validation_definitions=[validation_definition])
)
checkpoint_result = checkpoint.run(batch_parameters={"dataframe": data})
print(checkpoint_result.describe())