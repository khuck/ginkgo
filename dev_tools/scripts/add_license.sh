#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
# ${THIS_DIR} is expected to be: ${GINKGO_ROOT_DIR}/dev_tools/scripts

# Use local paths, so there is less chance of a newline being in a path of a found file
cd "${THIS_DIR}/../.." || exit
GINKGO_ROOT_DIR="."

GINKGO_LICENSE_BEGIN="// SPDX-FileCopyrightText:"
GINKGO_LICENSE_END="// SPDX-License-Identifier:"

# These two files are temporary files which will be created (and deleted).
# Therefore, the files should not already exist.
COMMENTED_LICENSE_FILE="${THIS_DIR}/commented_license.tmp"
DIFF_FILE="${THIS_DIR}/diff.patch.tmp"

# Test if required commands are present on the system:
if ! command -v find &> /dev/null; then
    echo 'The command `find` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
if ! command -v diff &> /dev/null; then
    echo 'The command `diff` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
if ! command -v patch &> /dev/null; then
    echo 'The command `patch` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
if ! command -v grep &> /dev/null; then
    echo 'The command `grep` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
if ! command -v sed &> /dev/null; then
    echo 'The command `sed` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
if ! command -v cut &> /dev/null; then
    echo 'The command `cut` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
if ! command -v date &> /dev/null; then
    echo 'The command `date` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi

CURRENT_YEAR=$(date +%Y)
echo "${GINKGO_LICENSE_BEGIN} 2017-${CURRENT_YEAR} The Ginkgo authors" > "$1"
echo "//" >> "$1"
echo "${GINKGO_LICENSE_END} BSD-3-Clause" >> "$1"

# Does not work if a found file (including the path) contains a newline
find "${GINKGO_ROOT_DIR}" \
    \( -name '*.cuh' -o -name '*.hpp' -o -name '*.hpp.in' -o -name '*.cpp' -o -name '*.cu' -o -name '*.hpp.inc' \) \
    -type f -print \
    | grep -F -v -f "${THIS_DIR}/add_license.ignore" \
    | \
    while IFS='' read -r i; do
        # `grep -F` is important here because the characters in the license should be matched against
        # and not interpreted as an expression.
        if ! grep -F -q -e "${GINKGO_LICENSE_BEGIN}" "${i}"
        then
            cat "${COMMENTED_LICENSE_FILE}" "${i}" >"${i}.new" && mv "${i}.new" "${i}"
        else
            beginning=$(grep -F -n -e "${GINKGO_LICENSE_BEGIN}" "${i}" | cut -d":" -f1)
            end=$(grep -F -n -e "${GINKGO_LICENSE_END}" "${i}" | cut -d":" -f1)
            end=$((end+1))
            diff -u <(sed -n "${beginning},${end}p" "${i}") "${COMMENTED_LICENSE_FILE}" > "${DIFF_FILE}"
            if [ "$(cat "${DIFF_FILE}")" != "" ]
            then
                patch "${i}" "${DIFF_FILE}"
            fi
            rm "${DIFF_FILE}"
        fi
    done

rm "${COMMENTED_LICENSE_FILE}"
