using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
using Object = UnityEngine.Object;
using Random = UnityEngine.Random;

public class SpaceManager : MonoBehaviour
{
    public GameObject meteorPrefab;
    public GameObject meteorsPosition;
    public GameObject meteorsObject;
    public GameObject mineralsObject;
    public GameObject mineralPrefab;
    private SphereCollider boundaryTrigger;
    public GameObject initMeteor;
    private Vector3 initMeteorPos;
    public int initMeteorsNumber;
    public SpaceShipController[] ships;
    
    // Start is called before the first frame update
    void Start()
    {
        boundaryTrigger = GetComponent<SphereCollider>();
        initMeteorPos = initMeteor.transform.position;
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

    public void MineralTaken(GameObject mineral, SpaceShipController ship)
    {
        float amount = mineral.GetComponent<MineralScript>().amount;
        ship.TakeMineral(amount);
        Destroy(mineral.gameObject);
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            SpaceShipController player = other.GetComponent<SpaceShipController>();
            player.AddCustomReward(-1.0f);
            player.EndEpisode();
        }
    }

    public Dictionary<String, object> GetObservations(int shipIdx)
    {
        Vector3 shipPosition = ships[shipIdx].transform.position;
        Dictionary<String, object> result = new Dictionary<String, object>();
        result["gate"] = GetComponentInChildren<GateScript>().transform.position - shipPosition;
        result["meteorsPos"] = meteorsPosition.transform.position - shipPosition;

        MeteorScript[] meteors = GetComponentsInChildren<MeteorScript>();
        result["meteorsNumber"] = (float)meteors.Length;
        List<Vector3> meteorsDists = new List<Vector3>();
        if (meteors.Length > 0)
        {
            IEnumerable<Vector3> meteorsPositions = meteors.Select(x => x.transform.position);
            IEnumerator<Vector3> meteorsDistances =
                meteorsPositions.OrderBy(x => Vector3.Distance(x, shipPosition)).GetEnumerator();
            meteorsDistances.MoveNext();
            int m = Mathf.Min(3, meteors.Length);
            while (m > 0)
            {
                Vector3 pos = meteorsDistances.Current;
                meteorsDists.Add(pos - shipPosition);
                meteorsDistances.MoveNext();
                m--;
            }
        }
        result["meteorsDists"] = meteorsDists;
        
        MineralScript[] minerals = GetComponentsInChildren<MineralScript>();
        result["mineralsNumber"] = (float)minerals.Length;
        List<Vector3> mineralDists = new List<Vector3>();
        if (minerals.Length > 0)
        {
            IEnumerable<Vector3> mineralsPositions = minerals.Select(x => x.transform.position);
            IEnumerator<Vector3> mineralsDistances =
                mineralsPositions.OrderBy(x => Vector3.Distance(x, shipPosition))
                    .GetEnumerator();
            mineralsDistances.MoveNext();
            int m = Mathf.Min(3, minerals.Length);
            while (m > 0)
            {
                Vector3 pos = mineralsDistances.Current;
                mineralDists.Add(pos - shipPosition);
                mineralsDistances.MoveNext();
                m--;
            }
        }

        int n = 3 - minerals.Length;
        if (n > 0)
        {
            while (n > 0)
            {
                mineralDists.Add(Vector3.zero);
                n--;
            }
        }
        result["mineralsDists"] = mineralDists;
        result["shipDist"] = GetMinDistanceToShips(shipIdx);

        return result;
    }

    private Vector3 GetMinDistanceToShips(int shipIdx)
    {
        Vector3 result = Vector3.zero;
        float minMag = float.PositiveInfinity;
        for (int i = 0; i < ships.Length; i++)
        {
            if (shipIdx != i)
            {
                Vector3 curDist = ships[i].transform.position - ships[shipIdx].transform.position;
                float curMag = curDist.magnitude;
                if (curMag < minMag)
                {
                    minMag = curMag;
                    result = curDist;
                }
            }
        }
        return result;
    }

    public void Reset()
    {
        MineralScript[] minerals = mineralsObject.GetComponentsInChildren<MineralScript>();
        foreach (var mn in minerals)
        {
            Destroy(mn.gameObject);
        }
        
        MeteorScript[] m = meteorsObject.GetComponentsInChildren<MeteorScript>();
        int add = initMeteorsNumber - m.Length;
        if (add > 0)
        {
            for (int i = 0; i < add; i++)
            {
                GenerateMeteor();
            }
        }

        MeteorScript m0 = m[0];
        m0.transform.position = initMeteorPos;
        MineralScript[] mins = mineralsObject.GetComponentsInChildren<MineralScript>();
        for (int i = 0; i < mins.Length; i++)
        {
            Destroy(mins[i].gameObject);
        }
    }

    public void EndEpisodeAll()
    {
        for (int i = 0; i < ships.Length; i++)
        {
            ships[i].EndEpisode();
        }
    }
}
