import argparse, os
import random
import logging
import json

try:
    import pymesh
except ModuleNotFoundError:
    pass

import tqdm

import proxies.utils as utils
import proxies.shapenet as shapenet
import proxies.logger as logger
from proxies.utils import GracefulProcessPoolExecutor
from proxies.utils import GracefulProcessPoolExecutorDebug


def validate_ops(ops, valid_ops):
    for op in ops:
        if op not in valid_ops:
            raise RuntimeError("Invalid operation {}".format(op))
    return ops


def try_sample(sample_list, sample_num):
    try:
        return random.sample(sample_list, sample_num)
    except ValueError:
        raise ValueError(
            "Requested too many samples, there are only {} objects".format(
                len(sample_list)
            )
        )


def main(
    input_dir,
    ops,
    threads,
    overwrite,
    num_breaks,
    class_subsample,
    instance_subsample,
    max_break,
    min_break,
    splits_file,
    num_renders,
    train_ratio,
    debug,
    feature_flag,
    break_all,
    outoforder,
    process_restoration,
    render_resolution,
    pointcloud_noise,
    no_normals,
):

    # The following names are reserved:
    # model_normalized.obj      <- the original shapenet model
    # model_waterproofed.obj    <- shapenet model, waterproofed
    # model_c.obj               <- shapenet model, smoothed and upsampled
    # model_b_x.obj             <- shapenet model, broken
    # model_r_x.obj             <- shapenet model, restoration

    logging.info("Performing the following operations: {}".format(ops))
    logging.info("Using {} thread(s)".format(threads))

    if (splits_file is not None) and os.path.exists(splits_file):
        logging.info("Loading saved data from splits file {}".format(splits_file))
        object_id_dict = json.load(open(splits_file, "r"))
        id_train_list = [
            shapenet.ShapenetObject(
                root=input_dir,
                class_id=o[0],
                object_id=o[1],
                resolution=render_resolution,
                noise=pointcloud_noise,
                normals=(not no_normals),
            )
            for o in object_id_dict["id_train_list"]
        ]
        id_test_list = [
            shapenet.ShapenetObject(
                root=input_dir,
                class_id=o[0],
                object_id=o[1],
                resolution=render_resolution,
                noise=pointcloud_noise,
                normals=(not no_normals),
            )
            for o in object_id_dict["id_test_list"]
        ]
        object_id_list = id_test_list + id_train_list

    else:
        # Obtain a list of all the objects in the shapenet dataset
        logging.info("Searching for shapenet objects ...")
        object_id_list = shapenet.shapenet_find_all(input_dir)
        logging.info("Found {} objects".format(len(object_id_list)))

        # Subsample this list, if required
        if (class_subsample is not None) or (instance_subsample is not None):
            class_list = list(set([o.class_id for o in object_id_list]))

            # Sample classes
            if class_subsample is not None:
                class_list = try_sample(class_list, class_subsample)

            # Group object by class
            id_by_class = []
            for c in class_list:
                id_by_class.append([o for o in object_id_list if o.class_id == c])
                print(c, len(id_by_class[-1]))

            # Sample instances
            if instance_subsample is not None:
                for idx in range(len(id_by_class)):
                    # It's often the case that
                    if len(id_by_class[idx]) < instance_subsample:
                        logging.warning(
                            "Only {} samples in class {}, adding all".format(
                                len(id_by_class[idx]), class_list[idx]
                            )
                        )
                    id_by_class[idx] = try_sample(
                        id_by_class[idx], min(instance_subsample, len(id_by_class[idx]))
                    )

            # Flatten list
            object_id_list = []
            for e in id_by_class:
                object_id_list.extend(e)

            id_train_list, id_test_list = utils.split_train_test(
                object_id_list, train_ratio
            )

        else:
            id_train_list, id_test_list = utils.split_train_test(
                object_id_list, train_ratio
            )

        logging.info("Reduced to {} objects after sampling".format(len(object_id_list)))

        # Save the list
        logging.info("Saving data to splits file {}".format(splits_file))
        json.dump(
            {
                "id_train_list": [o.get_id for o in id_train_list],
                "id_test_list": [o.get_id for o in id_test_list],
            },
            open(splits_file, "w"),
        )

        logging.info("Building subdirectories ...")
        for o in object_id_list:
            o.build_dirs()

    if outoforder:
        random.shuffle(id_train_list)
        random.shuffle(id_test_list)
        object_id_list = id_test_list + id_train_list

    logging.info("Processing {} objects".format(len(object_id_list)))
    logging.info(
        "{} train objects, {} test objects".format(
            len(id_train_list), len(id_test_list)
        )
    )
    logging.info("{} classes".format(len(set([o.class_id for o in object_id_list]))))

    global GracefulProcessPoolExecutor
    if (threads == 1) and (debug):
        # This will completely disable the pool
        GracefulProcessPoolExecutor = GracefulProcessPoolExecutorDebug

    with GracefulProcessPoolExecutor(max_workers=threads) as executor:
        try:
            if "WATERPROOF" in ops:
                import proxies.process_waterproof as process_waterproof

                logging.info("Waterproofing ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Get the paths
                    f_in = obj.path_model_normalized()
                    f_out = obj.path_model_waterproofed()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_waterproof.handsoff, f_in=f_in, f_out=f_out
                    )
                executor.graceful_finish()

            if "CLEAN" in ops:
                import proxies.process_normalize as process_normalize

                logging.info("Cleaning ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Get the paths
                    f_in = obj.path_model_waterproofed()
                    f_out = obj.path_model_c()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_normalize.process, f_in=f_in, f_out=f_out
                    )
                executor.graceful_finish()

            if "BREAK" in ops:
                import proxies.process_break as process_break

                logging.info("Breaking ...")

                if break_all:
                    pbar = tqdm.tqdm(object_id_list)
                else:
                    pbar = tqdm.tqdm(id_test_list)  # < Only break the test list

                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Get the paths
                    f_in = obj.path_model_c()

                    # Sumbit
                    for idx in range(num_breaks):
                        f_bro = obj.path_model_b(idx)
                        f_res = obj.path_model_r(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_bro) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_break.process,
                            f_in=f_in,
                            f_out=f_bro,
                            f_restoration=f_res,
                            validate=True,
                            save_meta=False,
                            min_break=min_break,
                            max_break=max_break,
                        )
                executor.graceful_finish()

            # Operations on complete, broken, restoration models can now be computed in parallel
            if "UPSAMPLE" in ops:
                import proxies.process_upsample as process_upsample

                logging.info("Upsampling ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    f_out = obj.path_model_c_upsampled()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_upsample.process, f_in=f_in, f_out=f_out
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_model_b_upsampled(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_upsample.process, f_in=f_in, f_out=f_out
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_model_r_upsampled(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_upsample.process, f_in=f_in, f_out=f_out
                            )

                executor.graceful_finish()

            # Operations on complete, broken, restoration models can now be computed in parallel
            if "TDF" in ops:
                import proxies.process_tdf as process_tdf

                logging.info("Getting TDF ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    f_out = obj.path_model_c_tdf()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_tdf.process, f_in=f_in, f_out=f_out
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_model_b_tdf(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_tdf.process, f_in=f_in, f_out=f_out
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_model_r_tdf(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_tdf.process, f_in=f_in, f_out=f_out
                            )

                executor.graceful_finish()

            # Operations on complete, broken, restoration models can now be computed in parallel
            if "SIGNED_TDF" in ops:
                import proxies.process_occupancies as process_occupancies

                logging.info("Getting TDF ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    f_out = obj.path_model_c_signed_tdf()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_occupancies.add_sign, f_in=f_in, f_out=f_out
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_model_b_signed_tdf(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_occupancies.add_sign, f_in=f_in, f_out=f_out
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_model_r_signed_tdf(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_occupancies.add_sign, f_in=f_in, f_out=f_out
                            )

                executor.graceful_finish()

            if "NOFRAC" in ops:
                import proxies.process_roughness as process_roughness

                logging.info("Removing fracture ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_model_b_nofrac(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_roughness.process, f_in=f_in, f_out=f_out
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_model_r_nofrac(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_roughness.process, f_in=f_in, f_out=f_out
                            )

                executor.graceful_finish()

            # 3D features
            if any([o in ops for o in ["SHOT", "PFH", "global_SHOT", "global_PFH"]]):
                import proxies.process_3d_features as process_3d

                logging.info("Computing 3D features ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    for op, path_handle, processor_handle in zip(
                        [
                            "SHOT",
                            "PFH",
                            "global_SHOT",
                            "global_PFH",
                            "global_SHOT",
                            "global_PFH",
                        ],
                        [
                            obj.path_feature_c_shot,
                            obj.path_feature_c_pfh,
                            obj.path_feature_c_global_grid_offset_shot,
                            obj.path_feature_c_global_grid_offset_pfh,
                            obj.path_feature_c_global_axis_cut_shot,
                            obj.path_feature_c_global_axis_cut_pfh,
                        ],
                        [
                            process_3d.process_SHOT,
                            process_3d.process_PFH,
                            process_3d.process_global_grid_offset_SHOT,
                            process_3d.process_global_grid_offset_PFH,
                            process_3d.process_global_axis_cuts_SHOT,
                            process_3d.process_global_axis_cuts_PFH,
                        ],
                    ):
                        if op in ops:
                            f_out = path_handle(flag=feature_flag)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                processor_handle,
                                f_in=f_in,
                                f_out=f_out,
                                grid_size=3,
                                axis_cuts=1,
                            )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        for op, path_handle, processor_handle in zip(
                            ["SHOT", "PFH", "global_SHOT", "global_PFH"],
                            [
                                obj.path_feature_b_shot,
                                obj.path_feature_b_pfh,
                                obj.path_feature_b_global_shot,
                                obj.path_feature_b_global_pfh,
                            ],
                            [
                                process_3d.process_SHOT,
                                process_3d.process_PFH,
                                process_3d.process_global_SHOT,
                                process_3d.process_global_PFH,
                            ],
                        ):
                            if op in ops:
                                f_out = path_handle(idx, flag=feature_flag)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    processor_handle, f_in=f_in, f_out=f_out
                                )

                        # Process broken
                        f_in = obj.path_model_b_nofrac(idx)
                        for op, path_handle, processor_handle in zip(
                            ["global_SHOT", "global_PFH"],
                            [
                                obj.path_feature_b_nofrac_global_shot,
                                obj.path_feature_b_nofrac_global_pfh,
                            ],
                            [
                                process_3d.process_global_SHOT,
                                process_3d.process_global_PFH,
                            ],
                        ):
                            if op in ops:
                                f_out = path_handle(idx, flag=feature_flag)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    processor_handle, f_in=f_in, f_out=f_out
                                )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            for op, path_handle, processor_handle in zip(
                                ["SHOT", "PFH", "global_SHOT", "global_PFH"],
                                [
                                    obj.path_feature_r_shot,
                                    obj.path_feature_r_pfh,
                                    obj.path_feature_r_global_shot,
                                    obj.path_feature_r_global_pfh,
                                ],
                                [
                                    process_3d.process_SHOT,
                                    process_3d.process_PFH,
                                    process_3d.process_global_SHOT,
                                    process_3d.process_global_PFH,
                                ],
                            ):
                                if op in ops:
                                    f_out = path_handle(idx, flag=feature_flag)
                                    if not os.path.exists(f_in) or (
                                        os.path.exists(f_out) and not overwrite
                                    ):
                                        continue
                                    executor.graceful_submit(
                                        processor_handle, f_in=f_in, f_out=f_out
                                    )

                            # Process restoration
                            f_in = obj.path_model_r_nofrac(idx)
                            for op, path_handle, processor_handle in zip(
                                ["global_SHOT", "global_PFH"],
                                [
                                    obj.path_feature_r_nofrac_global_shot,
                                    obj.path_feature_r_nofrac_global_pfh,
                                ],
                                [
                                    process_3d.process_global_SHOT,
                                    process_3d.process_global_PFH,
                                ],
                            ):
                                if op in ops:
                                    f_out = path_handle(idx, flag=feature_flag)
                                    if not os.path.exists(f_in) or (
                                        os.path.exists(f_out) and not overwrite
                                    ):
                                        continue
                                    executor.graceful_submit(
                                        processor_handle, f_in=f_in, f_out=f_out
                                    )

                executor.graceful_finish()

            if "global_POINTNET" in ops:
                import proxies.process_pointnet as pointnet

                logging.info("Extracting POINTNET ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    f_out = obj.path_feature_c_global_pointnet()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        pointnet.process,
                        f_in=f_in,
                        f_out=f_out,
                        sigma=obj.noise,
                        use_normals=obj.normals,
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_feature_b_global_pointnet(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            pointnet.process,
                            f_in=f_in,
                            f_out=f_out,
                            sigma=obj.noise,
                            use_normals=obj.normals,
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_feature_r_global_pointnet(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                pointnet.process,
                                f_in=f_in,
                                f_out=f_out,
                                sigma=obj.noise,
                                use_normals=obj.normals,
                            )

                executor.graceful_finish()

            if "FCGF" in ops:
                import proxies.process_fcgf as process_fcgf

                logging.info("Extracting FCGF ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    f_out = obj.path_feature_c_fcgf()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_fcgf.process, f_in=f_in, f_out=f_out
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_feature_b_fcgf(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_fcgf.process,
                            f_in=f_in,
                            f_out=f_out,
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_feature_r_fcgf(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_fcgf.process, f_in=f_in, f_out=f_out
                            )

                executor.graceful_finish()

            if "RENDER" in ops:
                import proxies.process_render as process_render

                logging.info("Rendering ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    for angle in range(0, 360, int(360 / num_renders)):
                        f_out = obj.path_render_c(angle)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_render.process,
                            f_in=f_in,
                            f_out=f_out,
                            angle=angle,
                            resolution=obj.resolution,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        for angle in range(0, 360, int(360 / num_renders)):
                            f_out = obj.path_render_b(angle, idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_render.process,
                                f_in=f_in,
                                f_out=f_out,
                                angle=angle,
                                resolution=obj.resolution,
                            )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            for angle in range(0, 360, int(360 / num_renders)):
                                f_out = obj.path_render_r(angle, idx)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    process_render.process,
                                    f_in=f_in,
                                    f_out=f_out,
                                    angle=angle,
                                    resolution=obj.resolution,
                                )

                executor.graceful_finish()

            if "VOXELIZE_ROT" in ops:
                import proxies.process_occupancies as process_occupancies

                logging.info("Rotating voxels ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    for angle in range(0, 360, int(360 / num_renders)):
                        f_out = obj.path_model_c_voxelized_rot(angle)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_occupancies.voxelize,
                            f_in=f_in,
                            f_out=f_out,
                            angle=angle,
                            save_as_mat=True,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        for angle in range(0, 360, int(360 / num_renders)):
                            f_out = obj.path_model_b_voxelized_rot(angle, idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_occupancies.voxelize,
                                f_in=f_in,
                                f_out=f_out,
                                angle=angle,
                                save_as_mat=True,
                            )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            for angle in range(0, 360, int(360 / num_renders)):
                                f_out = obj.path_model_r_voxelized_rot(angle, idx)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    process_occupancies.voxelize,
                                    f_in=f_in,
                                    f_out=f_out,
                                    angle=angle,
                                    save_as_mat=True,
                                )

                executor.graceful_finish()

            if "DEPTH" in ops:
                import proxies.process_depth as process_depth

                logging.info("Generating depth images ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    for angle in range(0, 360, int(360 / num_renders)):
                        f_out = obj.path_render_depth_c(angle)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_depth.process, f_in=f_in, f_out=f_out, angle=angle
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        for angle in range(0, 360, int(360 / num_renders)):
                            f_out = obj.path_render_depth_b(angle, idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_depth.process,
                                f_in=f_in,
                                f_out=f_out,
                                angle=angle,
                            )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            for angle in range(0, 360, int(360 / num_renders)):
                                f_out = obj.path_render_depth_r(angle, idx)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    process_depth.process,
                                    f_in=f_in,
                                    f_out=f_out,
                                    angle=angle,
                                )

                executor.graceful_finish()

            if "DEPTH_PANO" in ops:
                import proxies.process_depth_pano as process_depth_pano

                logging.info("Generating depth images ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    for angle in range(12):
                        f_out = obj.path_render_depth_pano_c(angle)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_depth_pano.process,
                            f_in=f_in,
                            f_out=f_out,
                            angle=angle,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        for angle in range(12):
                            f_out = obj.path_render_depth_pano_b(angle)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_depth_pano.process,
                                f_in=f_in,
                                f_out=f_out,
                                angle=angle,
                            )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            for angle in range(12):
                                f_out = obj.path_render_depth_pano_r(angle)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    process_depth_pano.process,
                                    f_in=f_in,
                                    f_out=f_out,
                                    angle=angle,
                                )

                executor.graceful_finish()

            if "VOXELIZE" in ops:
                import proxies.process_occupancies as process_occupancies

                logging.info("Generating depth images ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    f_in = obj.path_model_c()
                    f_out = obj.path_model_c_voxelized()
                    if not os.path.exists(f_in) or (
                        os.path.exists(f_out) and not overwrite
                    ):
                        continue
                    executor.graceful_submit(
                        process_occupancies.voxelize, f_in=f_in, f_out=f_out
                    )

                    for idx in range(num_breaks):
                        # Process broken
                        f_in = obj.path_model_b(idx)
                        f_out = obj.path_model_b_voxelized(idx)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_occupancies.voxelize, f_in=f_in, f_out=f_out
                        )

                        if process_restoration:
                            # Process restoration
                            f_in = obj.path_model_r(idx)
                            f_out = obj.path_model_r_voxelized(idx)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_occupancies.voxelize, f_in=f_in, f_out=f_out
                            )

                executor.graceful_finish()

            if "SIFT_PANO" in ops:
                import proxies.process_2d_features as process_2d

                logging.info("Generating depth images ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    for angle in range(12):
                        f_in = obj.path_render_depth_pano_c(angle)
                        f_out = obj.path_feature_c_sift_pano(angle)
                        if not os.path.exists(f_in) or (
                            os.path.exists(f_out) and not overwrite
                        ):
                            continue
                        executor.graceful_submit(
                            process_2d.process_eDSIFT,
                            f_in=f_in,
                            f_out=f_out,
                        )

                    for idx in range(num_breaks):
                        # Process broken
                        for angle in range(12):
                            f_in = obj.path_render_depth_pano_b(angle)
                            f_out = obj.path_feature_b_sift_pano(angle)
                            if not os.path.exists(f_in) or (
                                os.path.exists(f_out) and not overwrite
                            ):
                                continue
                            executor.graceful_submit(
                                process_2d.process_eDSIFT,
                                f_in=f_in,
                                f_out=f_out,
                            )

                        if process_restoration:
                            # Process restoration
                            for angle in range(12):
                                f_in = obj.path_render_depth_pano_r(angle)
                                f_out = obj.path_feature_r_sift_pano(angle)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    process_2d.process_eDSIFT,
                                    f_in=f_in,
                                    f_out=f_out,
                                )

                executor.graceful_finish()

            # Operations requiring renders can now be computed in parallel
            if any([o in ops for o in ["SIFT", "ORB", "global_VGG"]]):
                import proxies.process_2d_features as process_2d

                logging.info("Computing 2D features ...")

                pbar = tqdm.tqdm(object_id_list)
                for obj in pbar:
                    pbar.write(
                        "[{}]".format(os.path.dirname(obj.path_model_normalized()))
                    )

                    # Process complete
                    for angle in range(0, 360, int(360 / num_renders)):
                        f_in = obj.path_render_c(angle)
                        for op, path_handle, processor_handle in zip(
                            ["SIFT", "ORB", "global_VGG"],
                            [
                                obj.path_feature_c_sift,
                                obj.path_feature_c_orb,
                                obj.path_feature_c_global_vgg,
                            ],
                            [
                                process_2d.process_SIFT,
                                process_2d.process_ORB,
                                process_2d.process_VGG,
                            ],
                        ):
                            if op in ops:
                                f_out = path_handle(angle, flag=feature_flag)
                                if not os.path.exists(f_in) or (
                                    os.path.exists(f_out) and not overwrite
                                ):
                                    continue
                                executor.graceful_submit(
                                    processor_handle, f_in=f_in, f_out=f_out
                                )

                    for idx in range(num_breaks):
                        # Process broken
                        for angle in range(0, 360, int(360 / num_renders)):
                            f_in = obj.path_render_b(angle, idx)
                            for op, path_handle, processor_handle in zip(
                                ["SIFT", "ORB", "global_VGG"],
                                [
                                    obj.path_feature_b_sift,
                                    obj.path_feature_b_orb,
                                    obj.path_feature_b_global_vgg,
                                ],
                                [
                                    process_2d.process_SIFT,
                                    process_2d.process_ORB,
                                    process_2d.process_VGG,
                                ],
                            ):
                                if op in ops:
                                    f_out = path_handle(angle, idx, flag=feature_flag)
                                    if not os.path.exists(f_in) or (
                                        os.path.exists(f_out) and not overwrite
                                    ):
                                        continue
                                    executor.graceful_submit(
                                        processor_handle, f_in=f_in, f_out=f_out
                                    )

                            if process_restoration:
                                # Process restoration
                                f_in = obj.path_render_r(angle, idx)
                                for op, path_handle, processor_handle in zip(
                                    ["SIFT", "ORB", "VGG"],
                                    [
                                        obj.path_feature_r_sift,
                                        obj.path_feature_r_orb,
                                        obj.path_feature_r_global_vgg,
                                    ],
                                    [
                                        process_2d.process_SIFT,
                                        process_2d.process_ORB,
                                        process_2d.process_VGG,
                                    ],
                                ):
                                    if op in ops:
                                        f_out = path_handle(
                                            angle, idx, flag=feature_flag
                                        )
                                        if not os.path.exists(f_in) or (
                                            os.path.exists(f_out) and not overwrite
                                        ):
                                            continue
                                        executor.graceful_submit(
                                            processor_handle, f_in=f_in, f_out=f_out
                                        )

                executor.graceful_finish()

        except KeyboardInterrupt:
            logging.info("Waiting for running processes ...")
            executor.graceful_finish()

    # Print out any errors encountered
    if len(executor.exceptions_log) > 0:
        logging.info("SUMMARY: The following errors were encountered ...")
        for k, v in executor.exceptions_log.items():
            logging.info("{}: {}".format(k, v))
    else:
        logging.info("SUMMARY: All operations completed successfully.")


if __name__ == "__main__":
    valid_ops = [
        "WATERPROOF",
        "CLEAN",
        "BREAK",
        "RENDER",
        "DEPTH",
        "DEPTH_PANO",
        "FCGF",
        "SIFT",
        "SIFT_PANO",
        "ORB",
        "global_VGG",
        "global_POINTNET",
    ]

    parser = argparse.ArgumentParser(
        description="Applies a sequence of "
        + "transforms to all objects in a database in parallel. Upon "
        + "completion prints a summary of errors encountered during runtime."
    )
    parser.add_argument(
        dest="input",
        type=str,
        help="Location of the database. Pass the top level directory. For "
        + 'ShapeNet this would be "ShapeNet.v2". Models will be extracted '
        + "by name and by extension (ENSURE THERE ARE NO OTHER .obj FILES IN "
        + "THIS DIRECTORY).",
    )
    parser.add_argument(
        dest="splits",
        type=str,
        help=".json file path, this file will be created and will store the "
        + "ids of all objects in the training and testing split. Will be used "
        + "if preprocessing is restarted to accelerate initial steps.",
    )
    parser.add_argument(
        dest="ops",
        type=str,
        nargs="+",
        help="List of operations to apply. Possible operations are "
        + "{}.\n".format(valid_ops),
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads to use. This script uses multiprocessing so "
        + "it is not recommended to set this number higher than the number of "
        + "physical cores in your computer.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training samples to testing samples that will be saved "
        + "to the split file.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="If passed will overwrite existing files. Else will skip existing "
        + "files.",
    )
    parser.add_argument(
        "--breaks",
        "-b",
        type=int,
        default=1,
        help="Number of breaks to generate for each object. This will only be "
        + "used if BREAK is passed.",
    )
    parser.add_argument(
        "--break_all",
        default=False,
        action="store_true",
        help="If passed will break the train and test set. Else will only break "
        + "the test set.",
    )
    parser.add_argument(
        "--renders",
        "-r",
        type=int,
        default=8,
        help="Number of renders to generate for each object. This will only be "
        + "used if RENDER is passed.",
    )
    parser.add_argument(
        "--class_subsample",
        default=None,
        type=int,
        help="If passed, will randomly sample this many classes from the "
        + "dataset. Will override subsample flag.",
    )
    parser.add_argument(
        "--instance_subsample",
        default=None,
        type=int,
        help="If passed, will randomly sample this many instances from each "
        + "class from the dataset. Will override subsample flag.",
    )
    parser.add_argument(
        "--max_break",
        default=0.5,
        type=float,
        help="Max amount (percentage based) of the source model to remove in a "
        + "given break. Breaks will be retried if they remove more than this "
        + "amount.",
    )
    parser.add_argument(
        "--min_break",
        default=0.3,
        type=float,
        help="Min amount (percentage based) of the source model to remove in a "
        + "given break. Breaks will be retried if they remove less than this "
        + "amount.",
    )
    parser.add_argument(
        "--feature_flag",
        default="",
        type=str,
        help="If passed, will append a custom flag to the feature files. Use "
        + "this to save multiple versions of a feature without overwriting.",
    )
    parser.add_argument(
        "--outoforder",
        default=False,
        action="store_true",
        help="If passed, will shuffle the dataset before processing. Note this "
        + "will not alter the cotents of the split file. Use this option if "
        + "you plan to process data simultaneously with multiple scripts or PCS.",
    )
    parser.add_argument(
        "--restoration",
        default=False,
        action="store_true",
        help="If passed, will render and generate features for restoration as "
        + "well as broken objects.",
    )
    parser.add_argument(
        "--render_resolution",
        default=[640, 640],
        type=int,
        nargs="+",
        help="Renders will be this resolution.",
    )
    parser.add_argument(
        "--pointcloud_noise",
        default=0.0,
        type=float,
        help="Add gaussian noise to the point clouds.",
    )
    parser.add_argument(
        "--no_normals",
        default=False,
        action="store_true",
        help="Remove normals from the point clouds.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    validate_ops(args.ops, valid_ops)

    assert os.path.isdir(args.input), "Input directory does not exist."
    if args.input[-1] == "/":
        args.input = args.input[:-1]
    # assert "ShapeNetCore.v2" in os.path.basename(
    #     args.input
    # ), "Input dir should be ShapeNetCore.v2, was {}".format(args.input)

    main(
        args.input,
        args.ops,
        args.threads,
        args.overwrite,
        args.breaks,
        args.class_subsample,
        args.instance_subsample,
        args.max_break,
        args.min_break,
        args.splits,
        args.renders,
        args.train_ratio,
        args.debug,
        args.feature_flag,
        args.break_all,
        args.outoforder,
        args.restoration,
        args.render_resolution,
        args.pointcloud_noise,
        args.no_normals,
    )
