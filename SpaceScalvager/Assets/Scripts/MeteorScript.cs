using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeteorScript : MonoBehaviour
{
    public ParticleSystem explosion;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void TakeHit()
    {
        ParticleSystem expl = Instantiate(explosion, transform.position, Quaternion.identity);
        Destroy(expl.gameObject, 0.5f);
        SpaceManager spaceManager = GetComponentInParent<SpaceManager>();
        spaceManager.DestroyMeteor(gameObject);
    }
}
