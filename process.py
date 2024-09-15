import glob
import json
import os

import SimpleITK
import torch

from predict_seg import PredictModel

from predict_cls import generate_mip, trace_classification


class Datacentric_baseline:  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces (Automated PET/CT lesion segmentation)
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        # according to the specified grand-challenge interfaces (Data centric model)
        self.output_path_category = "/output/data-centric-model.json"
        # where to store the nii files
        self.nii_path     = "/opt/algorithm/"
        self.weights_path = "/opt/algorithm/weights/"
        self.result_path  = "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)


        self.classfication_json = "/output/output_cls.json"
        self.mip_path   = "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/mipTs"
        self.trace_cls_modelfile  = os.path.join(self.weights_path, 'best.pt')

        self.ckpt_paths_fdg  = glob.glob(os.path.join(self.weights_path, "fdg*.ckpt"))
        self.ckpt_paths_psma = glob.glob(os.path.join(self.weights_path, "psma*.ckpt"))
        self.tta             = False
        self.sw_batch_size   = 6
        self.random_flips    = 1
        self.dynamic_tta     = False
        self.max_tta_time    = 220
        self.pet_percentiles_max_psma = 280
        
        self.inferer_fdg = PredictModel(
            model_paths=self.ckpt_paths_fdg,
            sw_batch_size=self.sw_batch_size,
            tta=self.tta,
            random_flips=self.random_flips,
            dynamic_tta=self.dynamic_tta,
            max_tta_time=self.max_tta_time,
        )
        self.inferer_psma = PredictModel(
            model_paths=self.ckpt_paths_psma,
            sw_batch_size=self.sw_batch_size,
            tta=self.tta,
            random_flips=self.random_flips,
            dynamic_tta=self.dynamic_tta,
            max_tta_time=self.max_tta_time,
            pet_percentiles_max=self.pet_percentiles_max_psma
        )

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def save_datacentric(self, value: bool):
        print("Saving datacentric json to " + self.output_path_category)
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mhas = os.listdir(os.path.join(self.input_path, "images/ct/"))
        pet_mhas = os.listdir(os.path.join(self.input_path, "images/pet/"))
        uuids = []
        for ct_mha in ct_mhas:
            uuids.append(os.path.splitext(ct_mha)[0])

        subj = []
        for n, ct_mha in enumerate(ct_mhas):
            sub_name = f"TCIA_{str(n).zfill(3)}"
            self.convert_mha_to_nii(
                os.path.join(self.input_path, "images/ct/", ct_mha),
                os.path.join(self.nii_path, f"{sub_name}_0000.nii.gz"),
            )
            self.convert_mha_to_nii(
                os.path.join(self.input_path, "images/pet/", pet_mhas[n]),
                os.path.join(self.nii_path, f"{sub_name}_0001.nii.gz"),
            )
            subj.append(sub_name)

        self.subj = subj
        self.uuid = uuids

        return uuids, subj

    def write_outputs(self, uuids, subjs):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        for u, uuid in enumerate(uuids):
            print(subjs[u])
            self.convert_nii_to_mha(
                os.path.join(self.result_path, subjs[u] + ".nii.gz"),
                os.path.join(self.output_path, uuid + ".mha"),
            )
            print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))


    def predict(self):
        """
        Your algorithm goes here
        """
        print("Trace classification starting!")
        generate_mip(self.nii_path, self.subj, self.mip_path, multithread=True, num_threads=10)
        fdg_subj, psma_subj = trace_classification(self.trace_cls_modelfile,
                                                   self.mip_path,
                                                   json_file=self.classfication_json)


        print("\n\nnnUNet segmentation starting!")
        print('\n', '='*5, 'FDG SEGMENTATION')
        print("Using weights: ", self.ckpt_paths_fdg)
        for sub in fdg_subj:
            self.inferer_fdg.run(
                ct_file_path  = os.path.join(self.nii_path, f"{sub}_0000.nii.gz"),
                pet_file_path = os.path.join(self.nii_path, f"{sub}_0001.nii.gz"),
                save_path     = self.result_path,
                subid         = sub,
                verbose       = True,
            )

        print('\n', '='*5, 'PSMA SEGMENTATION')
        print("Using weights: ", self.ckpt_paths_psma)
        for sub in psma_subj:
            self.inferer_psma.run(
                ct_file_path  = os.path.join(self.nii_path, f"{sub}_0000.nii.gz"),
                pet_file_path = os.path.join(self.nii_path, f"{sub}_0001.nii.gz"),
                save_path     = self.result_path,
                subid         = sub,
                verbose       = True,
            )

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()

        print("Start processing")
        uuid, subj = self.load_inputs()

        print("Start prediction")
        self.predict()

        print("Start output writing")
        self.save_datacentric(True)
        self.write_outputs(uuid, subj)


if __name__ == "__main__":
    Datacentric_baseline().process()
