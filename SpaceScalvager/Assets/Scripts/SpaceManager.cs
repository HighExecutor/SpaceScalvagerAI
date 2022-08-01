using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class SpaceManager : MonoBehaviour
{
    public GameObject meteorPrefab;
    public GameObject meteorsPosition;
    public GameObject meteorsObject;
    public GameObject mineralsObject;
    public GameObject mineralPrefab;
    private SpaceShipController player;
    
    // Start is called before the first frame update
    void Start()
    {
        player = GetComponentInChildren<SpaceShipController>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    
    public void GenerateMeteor()
    {
        Vector3 pos = Vector3.zero;
        pos.x = meteorsPosition.transform.position.x + Random.Range(-10.0f, 10.0f);
        pos.y = meteorsPosition.transform.position.y + Random.Range(-10.0f, 10.0f);
        pos.z = meteorsPosition.transform.position.z + Random.Range(-10.0f, 10.0f);
        GameObject meteor = Instantiate(meteorPrefab, pos, Quaternion.identity);
        meteor.transform.SetParent(meteorsObject.transform);
    }

    public void DestroyMeteor(GameObject meteor)
    {
        Destroy(meteor.gameObject);
        GameObject mineral = Instantiate(mineralPrefab, meteor.transform.position, Quaternion.identity);
        mineral.transform.SetParent(mineralsObject.transform);
        GenerateMeteor();
    }

    public void MineralTaken(GameObject mineral)
    {
        float amount = mineral.GetComponent<MineralScript>().amount;
        player.TakeMineral(amount);
        Destroy(mineral.gameObject);
    }

}
