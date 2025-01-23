// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract StorageContract is AccessControl {
    bytes32 public constant DATA_MANAGER_ROLE = keccak256("DATA_MANAGER_ROLE");

    struct DataRecord {
        uint256 id;
        string dataHash;
        uint256 timestamp;
        address uploader;
        string metadata;
    }

    // Mapping from record ID to DataRecord
    mapping(uint256 => DataRecord) private records;

    // Total number of records
    uint256 private recordCount;

    // Events
    event DataStored(uint256 indexed id, string dataHash, address indexed uploader, string metadata);

    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(DATA_MANAGER_ROLE, msg.sender);
    }

    /**
     * @dev Stores a new data record.
     * @param dataHash The hash of the data being stored.
     * @param metadata Additional metadata about the data.
     */
    function storeData(string calldata dataHash, string calldata metadata) external onlyRole(DATA_MANAGER_ROLE) {
        require(bytes(dataHash).length > 0, "Data hash cannot be empty");

        recordCount += 1;
        records[recordCount] = DataRecord({
            id: recordCount,
            dataHash: dataHash,
            timestamp: block.timestamp,
            uploader: msg.sender,
            metadata: metadata
        });

        emit DataStored(recordCount, dataHash, msg.sender, metadata);
    }

    /**
     * @dev Retrieves a data record by ID.
     * @param id The ID of the data record.
     * @return The DataRecord struct.
     */
    function getData(uint256 id) external view returns (DataRecord memory) {
        require(id > 0 && id <= recordCount, "Invalid record ID");
        return records[id];
    }

    /**
     * @dev Returns the total number of data records stored.
     */
    function getTotalRecords() external view returns (uint256) {
        return recordCount;
    }

    /**
     * @dev Grants DATA_MANAGER_ROLE to a new account.
     * @param account The address to grant the role.
     */
    function grantDataManagerRole(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        grantRole(DATA_MANAGER_ROLE, account);
    }

    /**
     * @dev Revokes DATA_MANAGER_ROLE from an account.
     * @param account The address to revoke the role.
     */
    function revokeDataManagerRole(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        revokeRole(DATA_MANAGER_ROLE, account);
    }
}

