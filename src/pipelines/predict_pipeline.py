import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts",'model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            # Check if all required columns are present in the input data
            required_columns = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
            print("Required columns :",required_columns)
            print("Feature columns:",features.columns)
            missing_columns = set(required_columns) - set(features.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in dataset: {missing_columns}")
            
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
                InteriorsStyle : object,
                PriceIndex8,
                ListDate, 
                Material,
                PriceIndex9,
                Agency,
                AreaIncomeType,
                EnvRating,
                PriceIndex7,
                ExpeditedListing, 
                PriceIndex4, PriceIndex1, PriceIndex6,
                PRIMEUNIT, Channel, Zip, InsurancePremiumIndex, PlotType,
                Architecture, PriceIndex3, Region, PriceIndex5, SubModel,
                Facade, State, NormalisedPopulation, BuildYear, RegionType,
                PropertyAge, PriceIndex2):

                self.InteriorsStyle = InteriorsStyle
                self.PriceIndex8 = PriceIndex8
                self.ListDate = ListDate
                self.Material = Material
                self.PriceIndex9 = PriceIndex9
                self.Agency= Agency
                self.AreaIncomeType = AreaIncomeType
                self.EnvRating = EnvRating
                self.PriceIndex7 = PriceIndex7
                self.ExpeditedListing = ExpeditedListing
                self.PriceIndex4 = PriceIndex4
                self.PriceIndex1 = PriceIndex1
                self.PriceIndex6 = PriceIndex6
                self.PRIMEUNIT = PRIMEUNIT
                self.Channel = Channel
                self.Zip = Zip
                self.InsurancePremiumIndex = InsurancePremiumIndex
                self.PlotType = PlotType
                self.Architecture = Architecture 
                self.PriceIndex3 = PriceIndex3
                self.Region = Region
                self.PriceIndex5 = PriceIndex5
                self.SubModel = SubModel
                self.Facade = Facade
                self.State = State
                self.NormalisedPopulation = NormalisedPopulation
                self.BuildYear = BuildYear
                self.RegionType = RegionType
                self.PropertyAge = PropertyAge
                self.PriceIndex2 = PriceIndex2

            

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "InteriorsStyle":[self.InteriorsStyle], 
                "PriceIndex8":[self.PriceIndex8],
                "ListDate":[self.ListDate],
                "Material":[self.Material],
                "PriceIndex9":[self.PriceIndex9], 
                "Agency":[self.Agency], 
                "AreaIncomeType":[self.AreaIncomeType], 
                "EnvRating":[self.EnvRating], 
                "PriceIndex7":[self.PriceIndex7],
                "ExpeditedListing":[self.ExpeditedListing], 
                "PriceIndex4":[self.PriceIndex4], 
                "PriceIndex1":[self.PriceIndex1], 
                "PriceIndex6":[self.PriceIndex6],
                "PRIMEUNIT":[self.PRIMEUNIT],
                "Channel":[self.Channel],
                "Zip":[self.Zip],
                "InsurancePremiumIndex":[self.InsurancePremiumIndex], 
                "PlotType":[self.PlotType],
                "Architecture":[self.Architecture], 
                "PriceIndex3":[self.PriceIndex3], 
                "Region":[self.Region], 
                "PriceIndex5":[self.PriceIndex5], 
                "SubModel":[self.SubModel],
                "Facade":[self.Facade],  
                "State":[self.State], 
                "NormalisedPopulation":[self.NormalisedPopulation], 
                "BuildYear":[self.BuildYear], 
                "RegionType":[self.RegionType],
                "PropertyAge":[self.PropertyAge], 
                "PriceIndex2":[self.PriceIndex2]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)