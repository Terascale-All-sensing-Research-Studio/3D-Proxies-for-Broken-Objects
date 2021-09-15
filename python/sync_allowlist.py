import os, argparse
import logging
import json

from proxies.database import ObjectDatabase
import proxies.logger as logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Allowlists make sure that"
        + "you pass the same objects to all models you test."
    )
    parser.add_argument(
        "--indices",
        type=str,
        nargs="+",
        help="list of index databases to sync.",
    )
    parser.add_argument(
        "--allowlist",
        type=str,
        default=None,
        help="Use to make sure results are consistent between different approaches.",
    )
    parser.add_argument(
        "--sync_vox_depth",
        default=False,
        action="store_true",
        help="If passed will also sync voxel/depth images.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    assert args.allowlist is not None

    if not isinstance(args.indices, list):
        args.indices = [args.indices]

    for idx, load_from in enumerate(args.indices):
        # Load the database
        odb = ObjectDatabase(load_from=load_from)
        print("Pruning from: {}".format(load_from))

        allow_list_file = args.allowlist

        # Create allowlist for the first time
        if idx == 0:
            logging.info("Building allowlist: {}".format(allow_list_file))

            database_allow_list = []
            for o in odb.database_objects:
                o_full_id = o.get_id_str
                database_allow_list.append(o_full_id)
            query_allow_list = []
            for o in odb.query_objects:
                o_full_id = o.get_id_str
                query_allow_list.append(o_full_id)

            print("{} database objects".format(len(database_allow_list)))
            print("{} query objects".format(len(query_allow_list)))

        # Add to the allowlist
        else:
            logging.info("Adding to allowlist: {}".format(allow_list_file))

            new_database_allow_list = []
            for o in odb.database_objects:
                o_full_id = o.get_id_str
                if o_full_id in database_allow_list:
                    new_database_allow_list.append(o_full_id)
            new_query_allow_list = []
            for o in odb.query_objects:
                o_full_id = o.get_id_str
                if o_full_id in query_allow_list:
                    new_query_allow_list.append(o_full_id)

            database_allow_list = new_database_allow_list
            query_allow_list = new_query_allow_list

            print(
                "{} database objects removed".format(
                    len(odb.database_objects) - len(database_allow_list)
                )
            )
            print(
                "{} query objects removed".format(
                    len(odb.query_objects) - len(query_allow_list)
                )
            )

    # Sync vox depth
    if args.sync_vox_depth:
        print("Pruning from: Voxel_Depth")
        new_database_allow_list = []
        for o in odb.database_objects:
            o_full_id = o.get_id_str
            if o_full_id in database_allow_list:
                if os.path.exists(o.path_model_c_signed_tdf()):
                    new_database_allow_list.append(o_full_id)

        new_query_allow_list = []
        for o in odb.query_objects:
            o_full_id = o.get_id_str
            if o_full_id in query_allow_list:
                if os.path.exists(o.path_model_b_signed_tdf()):
                    new_query_allow_list.append(o_full_id)

        print(
            "{} database objects removed".format(
                len(database_allow_list) - len(new_database_allow_list)
            )
        )
        print(
            "{} query objects removed".format(
                len(query_allow_list) - len(new_query_allow_list)
            )
        )

        database_allow_list = new_database_allow_list
        query_allow_list = new_query_allow_list

    # Write to disk
    json.dump(
        {
            "database": database_allow_list,
            "query": query_allow_list,
        },
        open(allow_list_file, "w"),
    )
